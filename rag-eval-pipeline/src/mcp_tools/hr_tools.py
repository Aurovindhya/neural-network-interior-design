from fastmcp import FastMCP
from src.retrieval.vector_store import retrieve
from typing import Any
import json

mcp_hr = FastMCP("HealthAgent HR Tools")

# Mock HR data — replace with real DB calls
HR_DATA = {
    "EMP001": {
        "name": "Sarah Chen",
        "department": "Nursing",
        "role": "Senior RN",
        "hire_date": "2019-03-15",
        "certifications": [
            {"name": "BLS", "expiry": "2025-06-01", "status": "active"},
            {"name": "ACLS", "expiry": "2024-11-15", "status": "expired"},
        ],
        "manager": "Dr. Patel",
        "status": "active",
    },
    "EMP002": {
        "name": "Marcus Webb",
        "department": "Radiology",
        "role": "Radiologic Technologist",
        "hire_date": "2021-07-20",
        "certifications": [
            {"name": "ARRT", "expiry": "2026-01-01", "status": "active"},
        ],
        "manager": "Dr. Nguyen",
        "status": "active",
    },
}

WORKFORCE_DATA = {
    "Nursing": {"headcount": 42, "open_roles": 3, "avg_tenure_years": 4.2},
    "Radiology": {"headcount": 12, "open_roles": 1, "avg_tenure_years": 3.8},
    "Administration": {"headcount": 18, "open_roles": 0, "avg_tenure_years": 6.1},
    "Emergency": {"headcount": 28, "open_roles": 5, "avg_tenure_years": 2.9},
}


@mcp_hr.tool()
def get_employee_profile(employee_id: str) -> dict[str, Any]:
    """Fetch a staff member's full profile including role, department, and hire date."""
    employee = HR_DATA.get(employee_id.upper())
    if not employee:
        return {"error": f"No employee found with ID {employee_id}"}
    return employee


@mcp_hr.tool()
def check_compliance_status(employee_id: str) -> dict[str, Any]:
    """Check whether an employee's certifications and licenses are current or expired."""
    employee = HR_DATA.get(employee_id.upper())
    if not employee:
        return {"error": f"No employee found with ID {employee_id}"}

    certs = employee.get("certifications", [])
    expired = [c for c in certs if c["status"] == "expired"]
    active = [c for c in certs if c["status"] == "active"]

    return {
        "employee_id": employee_id,
        "employee_name": employee["name"],
        "compliant": len(expired) == 0,
        "active_certifications": active,
        "expired_certifications": expired,
        "action_required": len(expired) > 0,
    }


@mcp_hr.tool()
def get_workforce_insights(department: str = "all") -> dict[str, Any]:
    """Get staffing statistics and headcount insights for a department or the whole organisation."""
    if department.lower() == "all":
        return {
            "summary": WORKFORCE_DATA,
            "total_headcount": sum(v["headcount"] for v in WORKFORCE_DATA.values()),
            "total_open_roles": sum(v["open_roles"] for v in WORKFORCE_DATA.values()),
        }
    dept_data = WORKFORCE_DATA.get(department)
    if not dept_data:
        return {"error": f"Department '{department}' not found", "available": list(WORKFORCE_DATA.keys())}
    return {"department": department, **dept_data}


@mcp_hr.tool()
def query_hr_policy(question: str) -> dict[str, Any]:
    """Answer questions about HR policies using the organisation's policy document knowledge base."""
    docs, context = retrieve(question, collection_name="hr_policies")
    return {
        "context_retrieved": context,
        "source_chunks": len(docs),
        "question": question,
    }
