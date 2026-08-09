"""
KrishiDrishti Data Models
Repurposed from ScreenMind's screenmind/storage/models.py

DiagnosisRecord is the crop-disease equivalent of ScreenMind's ActivityRecord.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DiagnosisRecord(BaseModel):
    """
    Structured output from Gemma 4 crop analysis.
    Analogous to ScreenMind's ActivityRecord — represents one analysis event.
    """
    # Crop identification
    crop_name: str = Field(default="Unknown Crop", description="Identified crop species")
    crop_name_hindi: str = Field(default="", description="Crop name in Hindi")

    # Health status
    is_healthy: bool = Field(default=False, description="True if no disease detected")
    disease_detected: str = Field(default="Unknown", description="Disease name in English")
    disease_detected_hindi: str = Field(default="", description="Disease name in Hindi")

    # Severity assessment
    severity: str = Field(
        default="Unknown",
        description="Disease severity: None / Mild / Moderate / Severe"
    )
    affected_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Estimated percentage of plant affected"
    )

    # Symptom details
    symptoms_observed: List[str] = Field(
        default_factory=list,
        description="List of visible symptoms"
    )
    symptoms_hindi: List[str] = Field(
        default_factory=list,
        description="Symptoms in Hindi"
    )

    # Cause
    cause: str = Field(default="", description="Pathogen, pest, or deficiency causing the disease")

    # Treatment
    treatment_english: str = Field(default="", description="Treatment plan in English")
    treatment_hindi: str = Field(default="", description="Treatment plan in Hindi (उपचार)")

    # Prevention
    prevention_english: str = Field(default="", description="Prevention measures in English")
    prevention_hindi: str = Field(default="", description="Prevention measures in Hindi (बचाव)")

    # Urgency
    urgency: str = Field(
        default="Medium",
        description="Action urgency: Low / Medium / High / Critical"
    )

    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in the diagnosis (0.0–1.0)"
    )

    # Extra observations
    additional_notes: str = Field(
        default="",
        description="Additional observations about the plant/environment"
    )

    class Config:
        extra = "ignore"  # Silently drop unknown fields from Gemma's output


class DiagnosisHistoryEntry(BaseModel):
    """
    A diagnosis record stored in SQLite history.
    Wraps DiagnosisRecord with database metadata.
    Analogous to ScreenMind's ActivityRecord with timestamp in the DB.
    """
    id: int = Field(description="Auto-incremented database ID")
    timestamp: datetime = Field(description="When the diagnosis was performed")
    image_path: Optional[str] = Field(default=None, description="Path to stored image")
    farmer_note: Optional[str] = Field(default=None, description="Farmer's own description")
    analysis_mode: str = Field(default="balanced", description="Analysis mode used")
    diagnosis: DiagnosisRecord = Field(description="The full diagnosis result")

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    """Request model for API — farmer note attached to image upload."""
    farmer_note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional description of symptoms in farmer's own words"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Analysis mode override: fast / balanced / accurate"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    gemma_available: bool
    server: str
    version: str = "1.0.0"
    app: str = "KrishiDrishti"
