# KrishiDrishti - AI Crop Doctor

**AI-powered crop disease diagnosis. Upload a photo of your plant and Gemma 4 identifies the disease and prescribes treatment in Hindi and English.**

100% local. 100% private. Runs on 4GB VRAM. No cloud, no API keys, no data leaves your machine.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![Gemma 4](https://img.shields.io/badge/Gemma_4-Vision-8B5CF6?style=flat-square&logo=google)](https://ai.google.dev/gemma)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![License MIT](https://img.shields.io/badge/License-MIT-10B981?style=flat-square)](LICENSE)

---

## What it does

A farmer photographs a diseased crop leaf. KrishiDrishti sends the image to **Gemma 4 running locally** via llama-server. Gemma identifies:

- **Disease name** (English + Hindi)
- **Severity** (None / Mild / Moderate / Severe)
- **Affected area** percentage
- **Treatment plan** with specific fungicide/pesticide names available at Krishi Kendra
- **Prevention advice** for next season
- **Urgency level** (Low / Medium / High / Critical)

Everything is stored in a local SQLite database. History is searchable. Works offline after first model download.

---

## Built on ScreenMind

KrishiDrishti is built by repurposing **[ScreenMind](https://github.com/ayushh0110/ScreenMind)** (winner, Build with Gemma 4 hackathon). The core architecture is identical, only the domain changes:

| ScreenMind | KrishiDrishti |
|---|---|
| `engine/analyzer.py` - screenshot to activity JSON | `gemma_engine/crop_analyzer.py` - crop photo to diagnosis JSON |
| `engine/llm_client.py` - Gemma vision calls | `gemma_engine/llm_client.py` - same, minus audio |
| `storage/database.py` - activity history | `storage/database.py` - diagnosis history |
| `api/server.py` - timeline/chat endpoints | `api/server.py` - diagnose/history endpoints |
| Screen capture loop | Removed, replaced by photo upload |
| Meeting transcription | Removed, not needed for agriculture |

The same three-stage JSON parse pipeline (extract, repair, regex fallback) gives ~99% parse success rate on Gemma's structured output.

---

## Quick Start

### 1. Install dependencies

```bash
cd krishidrishti
pip install -e .
```

Or install directly:

```bash
pip install fastapi uvicorn httpx pillow python-multipart pydantic python-dotenv aiosqlite
```

### 2. Start llama-server with Gemma 4

Download a Gemma 4 GGUF model and start llama-server:

```bash
# Gemma 4 E2B (recommended, 4GB VRAM)
llama-server -m gemma-4-it-Q4_K_M.gguf --port 8080 --n-gpu-layers 99

# Gemma 4 E4B (6GB VRAM, better accuracy)
llama-server -m gemma-4-it-4B-Q4_K_M.gguf --port 8080 --n-gpu-layers 99

# CPU-only (slow but works)
llama-server -m gemma-4-it-Q4_K_M.gguf --port 8080
```

Download GGUF from: https://huggingface.co/google/gemma-4-it-GGUF

### 3. Run KrishiDrishti

```bash
python main.py
```

Opens `http://localhost:7878` in your browser automatically.

---

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
LLAMA_SERVER_HOST=http://localhost:8080
KRISHIDRISHTI_HOST=0.0.0.0
KRISHIDRISHTI_PORT=7878
ANALYSIS_MODE=balanced
DB_PATH=~/.krishidrishti/diagnoses.db
```

---

## Analysis Modes

| Mode | Time on 4GB GPU | When to use |
|---|---|---|
| `fast` | ~12s | Quick field check, many photos |
| `balanced` | ~30s | Default, best quality/speed tradeoff |
| `accurate` | ~60s | Difficult cases, multiple possible diseases |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/diagnose` | Upload crop photo and get diagnosis |
| `GET` | `/api/history` | Diagnosis history (paginated) |
| `GET` | `/api/history/{id}` | Single diagnosis detail |
| `DELETE` | `/api/history/{id}` | Delete a record |
| `GET` | `/api/search?q=wheat` | Search history by crop or disease |
| `GET` | `/api/stats` | Aggregate statistics |
| `GET` | `/api/health` | Gemma 4 connectivity status |
| `GET` | `/docs` | Interactive API documentation |

### Diagnose via curl

```bash
curl -X POST http://localhost:7878/api/diagnose \
  -F "image=@leaf_photo.jpg" \
  -F "farmer_note=pattiyoon par bhoore dhabbe hain" \
  -F "mode=balanced"
```

### Example response

```json
{
  "crop_name": "Wheat",
  "crop_name_hindi": "gehu",
  "is_healthy": false,
  "disease_detected": "Wheat Leaf Rust (Puccinia triticina)",
  "disease_detected_hindi": "Gehoon ka patti ka jang rog",
  "severity": "Moderate",
  "affected_percentage": 35,
  "symptoms_observed": [
    "Orange-brown pustules on leaves",
    "Yellow chlorotic areas"
  ],
  "cause": "Fungal, Puccinia triticina",
  "treatment_english": "Apply Propiconazole 25% EC at 0.1% or Tebuconazole 250 EW at 0.1% spray immediately.",
  "urgency": "High",
  "confidence": 0.89,
  "diagnosis_id": 42,
  "analysis_time_seconds": 28.4
}
```

---

## Project Structure

```
krishidrishti/
├── main.py                      Entry point
├── config.py                    Settings loaded from .env
├── pyproject.toml
├── .env.example
│
├── gemma_engine/
│   ├── llm_client.py            llama-server HTTP client (repurposed from ScreenMind)
│   └── crop_analyzer.py         Crop disease analysis engine (repurposed from ScreenMind)
│
├── storage/
│   ├── models.py                Pydantic models: DiagnosisRecord, DiagnosisHistoryEntry
│   └── database.py              Async SQLite with FTS5 search
│
└── api/
    ├── server.py                FastAPI REST endpoints
    └── static/
        └── index.html           Web UI (Hindi + English, drag-drop upload)
```

---

## Hardware Requirements

| Model | VRAM Needed | Quality |
|---|---|---|
| Gemma 4 E2B Q4 | 4 GB | Good |
| Gemma 4 E4B Q4 | 6 GB | Better |
| Gemma 4 12B Q4 | 10 GB | Best |
| CPU only | None | Slow (~5 min per image) |

Tested on: NVIDIA GTX 1650 (4GB), RTX 3060 (12GB), MacBook Air M1.

---

## CLI Options

```bash
python main.py --help

# Custom host/port
python main.py --host 0.0.0.0 --port 8000

# Override analysis mode
python main.py --mode accurate

# Skip opening browser
python main.py --no-browser
```

---

## Acknowledgements

Built on top of [ScreenMind](https://github.com/ayushh0110/ScreenMind) by [@ayushh0110](https://github.com/ayushh0110), the winning entry from the Build with Gemma 4 hackathon at AI Durg. The core Gemma inference architecture, JSON parse pipeline, and storage layer are repurposed from that project.

---

## License

MIT, free to use, modify, and distribute.
