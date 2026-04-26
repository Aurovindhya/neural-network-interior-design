from fastmcp import FastMCP
from src.retrieval.vector_store import retrieve
from typing import Any

mcp_patient = FastMCP("HealthAgent Patient Tools")

# Mock patient data — replace with real DB calls
PATIENT_DATA = {
    "PAT001": {
        "name": "James Okafor",
        "dob": "1985-04-12",
        "blood_type": "O+",
        "allergies": ["Penicillin", "Sulfa drugs"],
        "primary_physician": "Dr. Patel",
        "appointments": [
            {"date": "2025-05-10", "time": "09:30", "type": "Follow-up", "doctor": "Dr. Patel", "location": "Room 204"},
            {"date": "2025-06-02", "time": "14:00", "type": "Lab Results Review", "doctor": "Dr. Nguyen", "location": "Room 108"},
        ],
        "active_medications": [
            {"name": "Metformin", "dose": "500mg", "frequency": "twice daily", "with_food": True},
            {"name": "Lisinopril", "dose": "10mg", "frequency": "once daily", "with_food": False},
        ],
        "conditions": ["Type 2 Diabetes", "Hypertension"],
    },
    "PAT002": {
        "name": "Amara Singh",
        "dob": "1972-09-28",
        "blood_type": "A-",
        "allergies": [],
        "primary_physician": "Dr. Nguyen",
        "appointments": [
            {"date": "2025-05-15", "time": "11:00", "type": "Annual Checkup", "doctor": "Dr. Nguyen", "location": "Room 310"},
        ],
        "active_medications": [
            {"name": "Atorvastatin", "dose": "20mg", "frequency": "once daily at bedtime", "with_food": False},
        ],
        "conditions": ["High Cholesterol"],
    },
}


@mcp_patient.tool()
def get_upcoming_appointments(patient_id: str) -> dict[str, Any]:
    """Retrieve a patient's upcoming scheduled appointments."""
    patient = PATIENT_DATA.get(patient_id.upper())
    if not patient:
        return {"error": "Patient record not found"}
    return {
        "patient_name": patient["name"],
        "upcoming_appointments": patient["appointments"],
        "count": len(patient["appointments"]),
    }


@mcp_patient.tool()
def get_medical_summary(patient_id: str) -> dict[str, Any]:
    """Retrieve a high-level summary of a patient's medical record including conditions and allergies."""
    patient = PATIENT_DATA.get(patient_id.upper())
    if not patient:
        return {"error": "Patient record not found"}
    return {
        "name": patient["name"],
        "date_of_birth": patient["dob"],
        "blood_type": patient["blood_type"],
        "known_allergies": patient["allergies"],
        "active_conditions": patient["conditions"],
        "primary_physician": patient["primary_physician"],
    }


@mcp_patient.tool()
def get_medication_reminders(patient_id: str) -> dict[str, Any]:
    """Get a patient's current medications with dosage and timing instructions."""
    patient = PATIENT_DATA.get(patient_id.upper())
    if not patient:
        return {"error": "Patient record not found"}
    return {
        "patient_name": patient["name"],
        "medications": patient["active_medications"],
        "reminder": "Always consult your physician before adjusting any medication.",
    }


@mcp_patient.tool()
def query_care_plan(question: str) -> dict[str, Any]:
    """Answer patient questions about their care plan, conditions, or treatment using their care documents."""
    docs, context = retrieve(question, collection_name="care_plans")
    return {
        "context_retrieved": context,
        "source_chunks": len(docs),
        "question": question,
    }
