from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from src.config import get_settings
from src.auth import UserRole
from src.evaluation.langfuse_eval import get_langfuse_handler, evaluate_response
from src.mcp_tools.hr_tools import (
    get_employee_profile,
    check_compliance_status,
    get_workforce_insights,
    query_hr_policy,
)
from src.mcp_tools.patient_tools import (
    get_upcoming_appointments,
    get_medical_summary,
    get_medication_reminders,
    query_care_plan,
)
import uuid

settings = get_settings()

HR_SYSTEM_PROMPT = """You are HealthAgent, an intelligent assistant for healthcare HR teams.
You help HR staff manage employee records, check compliance, understand workforce data, and answer policy questions.
Always be precise, professional, and flag any compliance issues clearly.
When answering policy questions, cite the context retrieved from the knowledge base.
You only have access to data for employees within your organisation."""

PATIENT_SYSTEM_PROMPT = """You are HealthAgent, a caring and clear healthcare assistant for patients.
You help patients understand their appointments, medications, medical summaries, and care plans.
Always communicate in plain language — avoid medical jargon unless the patient uses it first.
Never provide a diagnosis or change a treatment recommendation. Always refer the patient to their physician for clinical decisions.
You only have access to the record of the logged-in patient."""


def build_hr_tools() -> list:
    return [
        StructuredTool.from_function(get_employee_profile, name="get_employee_profile"),
        StructuredTool.from_function(check_compliance_status, name="check_compliance_status"),
        StructuredTool.from_function(get_workforce_insights, name="get_workforce_insights"),
        StructuredTool.from_function(query_hr_policy, name="query_hr_policy"),
    ]


def build_patient_tools() -> list:
    return [
        StructuredTool.from_function(get_upcoming_appointments, name="get_upcoming_appointments"),
        StructuredTool.from_function(get_medical_summary, name="get_medical_summary"),
        StructuredTool.from_function(get_medication_reminders, name="get_medication_reminders"),
        StructuredTool.from_function(query_care_plan, name="query_care_plan"),
    ]


def create_agent(user: dict, session_id: str) -> AgentExecutor:
    role = user["role"]
    user_id = user.get("employee_id") or user.get("patient_id", "unknown")

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
    )

    tools = build_hr_tools() if role == UserRole.HR else build_patient_tools()
    system_prompt = HR_SYSTEM_PROMPT if role == UserRole.HR else PATIENT_SYSTEM_PROMPT

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_openai_tools_agent(llm, tools, prompt)

    langfuse_handler = get_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
        role=role.value,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        callbacks=[langfuse_handler],
        handle_parsing_errors=True,
    )


def run_agent(agent_executor: AgentExecutor, query: str, trace_id: str) -> dict:
    result = agent_executor.invoke({"input": query})
    response = result.get("output", "")

    scores = evaluate_response(
        trace_id=trace_id,
        query=query,
        response=response,
    )

    return {
        "response": response,
        "eval_scores": scores,
    }
