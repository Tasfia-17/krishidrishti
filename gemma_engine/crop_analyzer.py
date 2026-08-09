"""
KrishiDrishti — Gemma 4 Crop Disease Analyzer
Repurposed from ScreenMind's screenmind/engine/analyzer.py

Sends crop/plant photos to Gemma 4 via llama-server and returns structured
diagnosis data: disease identification, severity, treatment advice in Hindi and
English, confidence score, and actionable recommendations.

Three analysis modes (matching ScreenMind's fast/balanced/accurate pattern):
  - analyze_fast():     No thinking tokens, ~12s. For quick scans.
  - analyze_balanced(): Thinking enabled, ~30s. Best for demo.
  - analyze_accurate(): Thinking + multi-aspect analysis, ~60s. Maximum detail.
"""
import base64
import io
import json
import logging
import re
import time
from typing import Optional, Tuple

from PIL import Image

from config import settings
from gemma_engine import llm_client
from storage.models import DiagnosisRecord

logger = logging.getLogger("krishidrishti.gemma_engine.crop_analyzer")

# ── Prompt Engineering ────────────────────────────────────────────────────────
# The prompt is the core of KrishiDrishti. It instructs Gemma 4 to act as an
# agricultural expert and return structured JSON that our app can render.

SYSTEM_PROMPT = """You are KrishiDrishti, an expert AI agricultural scientist and plant pathologist.
You analyze photos of crops and plants to identify diseases, pests, and nutritional deficiencies.
You provide actionable treatment advice specifically relevant to Indian farmers.
Always respond with compassion and practical guidance that a rural farmer can act on immediately.
You must always respond in structured JSON format as instructed."""

CROP_ANALYSIS_PROMPT = """Analyze this crop/plant photo carefully. You are an expert plant pathologist helping an Indian farmer.

Return ONLY a valid JSON object with this exact structure:
{
  "crop_name": "identified crop (e.g., Wheat, Rice, Tomato, Cotton, Potato) or 'Unknown Crop'",
  "crop_name_hindi": "crop name in Hindi (e.g., गेहूं, चावल, टमाटर, कपास, आलू)",
  "is_healthy": true or false,
  "disease_detected": "exact disease name or 'None' if healthy",
  "disease_detected_hindi": "disease name in Hindi or 'कोई रोग नहीं' if healthy",
  "severity": "None / Mild / Moderate / Severe",
  "affected_percentage": estimated percentage of plant/leaf affected as integer (0-100),
  "symptoms_observed": ["list", "of", "visible", "symptoms"],
  "symptoms_hindi": ["लक्षणों की", "हिंदी में", "सूची"],
  "cause": "pathogen/pest/deficiency causing the disease (fungal, bacterial, viral, pest, nutritional deficiency, etc.)",
  "treatment_english": "Detailed treatment plan in 2-3 sentences with specific fungicide/pesticide names and application method",
  "treatment_hindi": "उपचार की विस्तृत जानकारी हिंदी में — 2-3 वाक्य, विशिष्ट दवाओं के नाम और उपयोग की विधि सहित",
  "prevention_english": "Prevention measures for the future in 1-2 sentences",
  "prevention_hindi": "भविष्य में बचाव के उपाय हिंदी में — 1-2 वाक्य",
  "urgency": "Low / Medium / High / Critical",
  "confidence": confidence score as float between 0.0 and 1.0,
  "additional_notes": "any other observations about crop health, soil, or environment visible in the image"
}

Rules:
- Be SPECIFIC about disease names (e.g., "Wheat Stem Rust (Puccinia graminis)" not just "rust")
- Recommend treatments available at Indian agricultural stores (Krishi Kendra)
- If the image is not a plant/crop, set is_healthy=false, disease_detected="Not a plant image", confidence=0.1
- If image is too blurry or unclear, note it in additional_notes and set confidence accordingly
- Always provide Hindi translation even for disease names
- Return ONLY valid JSON, nothing else."""


FAST_PROMPT = """Analyze this crop photo. Return ONLY valid JSON:
{
  "crop_name": "crop name",
  "crop_name_hindi": "हिंदी में फसल का नाम",
  "is_healthy": true or false,
  "disease_detected": "disease name or 'None'",
  "disease_detected_hindi": "रोग का नाम हिंदी में या 'कोई रोग नहीं'",
  "severity": "None / Mild / Moderate / Severe",
  "affected_percentage": 0,
  "symptoms_observed": ["symptom1", "symptom2"],
  "symptoms_hindi": ["लक्षण1"],
  "cause": "cause of disease",
  "treatment_english": "treatment plan with specific product names",
  "treatment_hindi": "हिंदी में उपचार",
  "prevention_english": "prevention advice",
  "prevention_hindi": "हिंदी में बचाव",
  "urgency": "Low / Medium / High / Critical",
  "confidence": 0.85,
  "additional_notes": "other observations"
}
Return ONLY valid JSON."""


class CropAnalyzer:
    """
    Analyzes crop/plant photos using Gemma 4 vision via llama-server.
    Repurposed from ScreenMind's GemmaAnalyzer — adapted for agriculture.

    The core logic (image encoding, prompt construction, JSON parsing,
    repair pipeline) is directly adapted from ScreenMind's analyzer.py.
    """

    def __init__(self):
        self._initialized = False

    def _ensure_client(self):
        """Verify llama-server is reachable before attempting inference."""
        if not self._initialized:
            if llm_client.is_available():
                self._initialized = True
                logger.info(f"Connected to llama-server at {settings.llama_server_host}")
            else:
                raise ConnectionError(
                    f"Cannot reach llama-server at {settings.llama_server_host}. "
                    "Make sure llama-server is running with a Gemma 4 model."
                )

    def analyze(
        self,
        image: Image.Image,
        farmer_note: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> DiagnosisRecord:
        """
        Analyze a crop image and return a structured DiagnosisRecord.

        Args:
            image: PIL Image of the crop/plant
            farmer_note: Optional text from the farmer describing symptoms
            mode: Override analysis mode (fast/balanced/accurate). Falls back to config.

        Returns:
            DiagnosisRecord with disease, severity, treatment, and confidence.
        """
        mode = mode or settings.analysis_mode
        if mode == "fast":
            return self._analyze_fast(image, farmer_note)
        elif mode == "accurate":
            return self._analyze_accurate(image, farmer_note)
        else:  # balanced (default)
            return self._analyze_balanced(image, farmer_note)

    def analyze_from_bytes(
        self,
        image_bytes: bytes,
        farmer_note: Optional[str] = None,
        mode: Optional[str] = None,
        content_type: str = "image/jpeg",
    ) -> DiagnosisRecord:
        """
        Analyze a crop image from raw bytes (e.g., file upload).
        Convenience method for the API layer.
        """
        image = Image.open(io.BytesIO(image_bytes))
        return self.analyze(image, farmer_note=farmer_note, mode=mode)

    def _analyze_balanced(
        self,
        image: Image.Image,
        farmer_note: Optional[str] = None,
    ) -> DiagnosisRecord:
        """
        Balanced mode: thinking enabled, ~30s.
        Best quality/speed tradeoff for the demo.
        """
        self._ensure_client()

        prompt = self._build_prompt(CROP_ANALYSIS_PROMPT, farmer_note)
        image_bytes = self._image_to_bytes(image)

        start = time.time()
        try:
            raw = llm_client.chat_with_image(
                prompt=prompt,
                image_bytes=image_bytes,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1200,
            )
            elapsed = time.time() - start
            logger.info(f"Balanced analysis done in {elapsed:.1f}s")
            return self._parse_response(raw, image)
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Balanced analysis error after {elapsed:.1f}s: {e}")
            return self._error_record(str(e))

    def _analyze_fast(
        self,
        image: Image.Image,
        farmer_note: Optional[str] = None,
    ) -> DiagnosisRecord:
        """
        Fast mode: no thinking tokens, ~12s.
        Uses JSON prefill trick (assistant pre-populated with '{') for speed.
        Directly adapted from ScreenMind's analyze_screenshot_fast().
        """
        self._ensure_client()

        prompt = self._build_prompt(FAST_PROMPT, farmer_note)
        image_bytes = self._image_to_bytes(image)
        img_b64 = base64.b64encode(image_bytes).decode()

        # Prefill the assistant response with '{' to force immediate JSON output
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
            {"role": "assistant", "content": "\n\n{"},
        ]

        for attempt in range(2):
            start = time.time()
            try:
                raw = llm_client.chat(
                    messages=messages,
                    temperature=0.0 if attempt == 0 else 0.1,
                    max_tokens=600,
                )
                # Prepend the '{' we used in the assistant prefill
                if not raw.strip().startswith("{"):
                    raw = "{" + raw
                elapsed = time.time() - start
                logger.info(f"Fast analysis done in {elapsed:.1f}s (attempt {attempt + 1})")

                record = self._safe_parse_json(raw, image)
                if record:
                    return record

                if attempt == 0:
                    logger.warning("Parse failed on fast mode, retrying...")
                    continue
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Fast analysis error, retrying: {e}")
                    continue
                logger.error(f"Fast analysis failed: {e}")

        return self._error_record("Analysis failed after retries")

    def _analyze_accurate(
        self,
        image: Image.Image,
        farmer_note: Optional[str] = None,
    ) -> DiagnosisRecord:
        """
        Accurate mode: full thinking enabled, ~60s.
        Best for difficult cases, multi-disease scenarios, or ambiguous images.
        """
        self._ensure_client()

        # Accurate mode gets a richer prompt asking for deeper analysis
        accurate_prompt = CROP_ANALYSIS_PROMPT + "\n\nIMPORTANT: Think carefully before answering. Consider all possible diseases that match the visible symptoms. If multiple diseases are possible, pick the most likely one but mention alternatives in additional_notes."

        prompt = self._build_prompt(accurate_prompt, farmer_note)
        image_bytes = self._image_to_bytes(image, max_dim=1024)  # Higher res for accuracy

        start = time.time()
        try:
            raw = llm_client.chat_with_image(
                prompt=prompt,
                image_bytes=image_bytes,
                system=SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1600,
            )
            elapsed = time.time() - start
            logger.info(f"Accurate analysis done in {elapsed:.1f}s")
            return self._parse_response(raw, image)
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Accurate analysis error after {elapsed:.1f}s: {e}")
            return self._error_record(str(e))

    # ── Image Processing ──────────────────────────────────────────────────────

    def _image_to_bytes(self, image: Image.Image, max_dim: int = 768) -> bytes:
        """
        Convert PIL Image to optimized JPEG bytes for Gemma 4 input.
        Adapted from ScreenMind's _image_to_bytes().

        768px balances accuracy vs VRAM usage on 4GB GPUs (Gemma 4 E2B).
        1024px for accurate mode where detail matters more.
        """
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    # ── Prompt Building ───────────────────────────────────────────────────────

    def _build_prompt(self, base_prompt: str, farmer_note: Optional[str]) -> str:
        """Append farmer's optional note to the base prompt."""
        if farmer_note and farmer_note.strip():
            return (
                f"{base_prompt}\n\n"
                f"Farmer's description of the problem: {farmer_note.strip()}"
            )
        return base_prompt

    # ── Response Parsing ──────────────────────────────────────────────────────
    # Adapted from ScreenMind's multi-stage parse pipeline.
    # Handles Gemma's tendency to wrap JSON in markdown, add thinking tokens,
    # or emit truncated/malformed JSON (~15% of the time).

    def _parse_response(self, raw: str, image: Optional[Image.Image] = None) -> DiagnosisRecord:
        """
        Main parse entry point.
        Strips thinking tags, extracts JSON, falls back to repair pipeline.
        """
        # Strip Gemma thinking tokens if present
        if "<think>" in raw and "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if "...done thinking." in raw:
            raw = raw.split("...done thinking.")[-1].strip()

        json_str = self._extract_json(raw)
        if json_str:
            try:
                data = json.loads(json_str)
                return self._build_record(data, image)
            except json.JSONDecodeError:
                pass

        # Repair pipeline
        record = self._safe_parse_json(raw, image)
        if record:
            return record

        logger.warning("All parse methods failed, returning error record")
        return self._error_record("Could not parse Gemma response")

    def _safe_parse_json(
        self, raw: str, image: Optional[Image.Image] = None
    ) -> Optional[DiagnosisRecord]:
        """
        Full parse pipeline: extract → parse → repair → regex.
        Returns DiagnosisRecord on success, None if everything fails.
        Adapted from ScreenMind's _safe_parse_json().
        """
        if not raw:
            return None

        json_str = self._extract_json(raw)
        if json_str:
            # Step 1: Direct parse
            try:
                data = json.loads(json_str)
                return self._build_record(data, image)
            except json.JSONDecodeError:
                pass
            except Exception:
                pass

            # Step 2: Repair and re-parse
            repaired = self._repair_json(json_str)
            if repaired:
                try:
                    data = json.loads(repaired)
                    logger.debug("JSON repaired successfully")
                    return self._build_record(data, image)
                except Exception:
                    pass

        # Step 3: Regex fallback
        fallback = self._regex_fallback(raw, image)
        if fallback:
            logger.debug("Used regex fallback for parsing")
            return fallback

        return None

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract a JSON object from raw Gemma output.
        Handles: clean JSON, markdown code blocks, JSON embedded in prose.
        Adapted from ScreenMind's _extract_json().
        """
        text = text.strip()

        # Already clean JSON
        if text.startswith("{") and text.endswith("}"):
            return text

        # JSON in markdown code block
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            return code_block.group(1)

        # Find first {...} block
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        # Deep match — any {...} including nested
        deep_match = re.search(r"\{.*\}", text, re.DOTALL)
        if deep_match:
            return deep_match.group(0)

        return None

    def _repair_json(self, broken: str) -> Optional[str]:
        """
        Attempt to fix common JSON issues from Gemma output.
        Handles: trailing commas, missing closing braces, truncated strings.
        Adapted from ScreenMind's _repair_json().
        """
        s = broken.strip()

        # Fix unescaped newlines inside strings
        s = re.sub(r'("(?:[^"\\]|\\.)*)"(\s*\n\s*)"', r'\1\\n\2', s)

        # Remove trailing commas before } or ]
        s = re.sub(r',\s*([}\]])', r'\1', s)

        # Fix Python-style booleans
        s = s.replace(": True", ": true").replace(": False", ": false")
        s = s.replace(":True", ": true").replace(":False", ": false")

        # Close open strings at end of truncated output
        if s.count('"') % 2 != 0:
            s += '"'

        # Balance braces
        open_braces = s.count('{') - s.count('}')
        open_brackets = s.count('[') - s.count(']')
        if open_brackets > 0:
            s += ']' * open_brackets
        if open_braces > 0:
            s += '}' * open_braces

        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            return None

    def _regex_fallback(
        self, text: str, image: Optional[Image.Image] = None
    ) -> Optional[DiagnosisRecord]:
        """
        Last-resort field extraction using regex patterns.
        Salvages individual fields even from heavily malformed responses.
        Adapted from ScreenMind's _regex_fallback().
        """
        def extract_str(pattern: str, default: str = "") -> str:
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else default

        def extract_bool(pattern: str, default: bool = False) -> bool:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).lower() in ("true", "1", "yes")
            return default

        def extract_float(pattern: str, default: float = 0.7) -> float:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            return default

        crop = extract_str(r'"crop_name"\s*:\s*"([^"]+)"', "Unknown Crop")
        disease = extract_str(r'"disease_detected"\s*:\s*"([^"]+)"', "Unknown")
        severity = extract_str(r'"severity"\s*:\s*"([^"]+)"', "Unknown")
        treatment_en = extract_str(r'"treatment_english"\s*:\s*"([^"]+)"', "Please consult your local Krishi Kendra.")
        treatment_hi = extract_str(r'"treatment_hindi"\s*:\s*"([^"]+)"', "कृपया अपने नजदीकी कृषि केंद्र से संपर्क करें।")
        confidence = extract_float(r'"confidence"\s*:\s*([0-9.]+)', 0.4)
        is_healthy = extract_bool(r'"is_healthy"\s*:\s*(true|false)', False)

        if crop == "Unknown Crop" and disease == "Unknown":
            return None  # Nothing useful extracted

        return DiagnosisRecord(
            crop_name=crop,
            crop_name_hindi=extract_str(r'"crop_name_hindi"\s*:\s*"([^"]+)"', crop),
            is_healthy=is_healthy,
            disease_detected=disease,
            disease_detected_hindi=extract_str(r'"disease_detected_hindi"\s*:\s*"([^"]+)"', disease),
            severity=severity,
            affected_percentage=0,
            symptoms_observed=[],
            symptoms_hindi=[],
            cause=extract_str(r'"cause"\s*:\s*"([^"]+)"', ""),
            treatment_english=treatment_en,
            treatment_hindi=treatment_hi,
            prevention_english=extract_str(r'"prevention_english"\s*:\s*"([^"]+)"', ""),
            prevention_hindi=extract_str(r'"prevention_hindi"\s*:\s*"([^"]+)"', ""),
            urgency=extract_str(r'"urgency"\s*:\s*"([^"]+)"', "Medium"),
            confidence=confidence,
            additional_notes="(Parsed via regex fallback — partial data)",
        )

    def _build_record(
        self, data: dict, image: Optional[Image.Image] = None
    ) -> DiagnosisRecord:
        """
        Construct a DiagnosisRecord from parsed JSON data.
        Normalizes fields and applies defaults for missing values.
        """
        # Normalize severity
        severity = str(data.get("severity", "Unknown")).strip()
        if severity.lower() in ("none", "healthy", "0"):
            severity = "None"
        elif severity.lower() not in ("mild", "moderate", "severe", "none"):
            severity = "Unknown"

        # Normalize urgency
        urgency = str(data.get("urgency", "Medium")).strip()
        if urgency.lower() not in ("low", "medium", "high", "critical"):
            urgency = "Medium"
        else:
            urgency = urgency.capitalize()

        # Clamp confidence
        try:
            confidence = float(data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        # Normalize affected_percentage
        try:
            affected = int(data.get("affected_percentage", 0))
            affected = max(0, min(100, affected))
        except (TypeError, ValueError):
            affected = 0

        # Ensure list fields are actually lists
        symptoms_observed = data.get("symptoms_observed", [])
        if isinstance(symptoms_observed, str):
            symptoms_observed = [symptoms_observed]

        symptoms_hindi = data.get("symptoms_hindi", [])
        if isinstance(symptoms_hindi, str):
            symptoms_hindi = [symptoms_hindi]

        return DiagnosisRecord(
            crop_name=str(data.get("crop_name", "Unknown Crop")),
            crop_name_hindi=str(data.get("crop_name_hindi", "")),
            is_healthy=bool(data.get("is_healthy", False)),
            disease_detected=str(data.get("disease_detected", "Unknown")),
            disease_detected_hindi=str(data.get("disease_detected_hindi", "")),
            severity=severity,
            affected_percentage=affected,
            symptoms_observed=symptoms_observed,
            symptoms_hindi=symptoms_hindi,
            cause=str(data.get("cause", "")),
            treatment_english=str(data.get("treatment_english", "")),
            treatment_hindi=str(data.get("treatment_hindi", "")),
            prevention_english=str(data.get("prevention_english", "")),
            prevention_hindi=str(data.get("prevention_hindi", "")),
            urgency=urgency,
            confidence=confidence,
            additional_notes=str(data.get("additional_notes", "")),
        )

    def _error_record(self, error_msg: str) -> DiagnosisRecord:
        """Return a DiagnosisRecord indicating analysis failure."""
        return DiagnosisRecord(
            crop_name="Unknown",
            crop_name_hindi="अज्ञात",
            is_healthy=False,
            disease_detected=f"Analysis Error",
            disease_detected_hindi="विश्लेषण विफल",
            severity="Unknown",
            affected_percentage=0,
            symptoms_observed=[],
            symptoms_hindi=[],
            cause="",
            treatment_english="Please retake the photo and try again. Ensure good lighting and the affected area is clearly visible.",
            treatment_hindi="कृपया दोबारा फोटो लें और पुनः प्रयास करें। सुनिश्चित करें कि रोशनी अच्छी हो और प्रभावित क्षेत्र साफ दिखे।",
            prevention_english="",
            prevention_hindi="",
            urgency="Low",
            confidence=0.0,
            additional_notes=f"Error: {error_msg[:200]}",
        )

    def is_available(self) -> bool:
        """Check if llama-server is reachable."""
        return llm_client.is_available()
