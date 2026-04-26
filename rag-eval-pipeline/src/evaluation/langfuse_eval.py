from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from src.config import get_settings
from typing import Optional
import time

settings = get_settings()

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)


def get_langfuse_handler(session_id: str, user_id: str, role: str) -> CallbackHandler:
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        session_id=session_id,
        user_id=user_id,
        metadata={"role": role},
    )


def evaluate_response(
    trace_id: str,
    query: str,
    response: str,
    context: Optional[str] = None,
) -> dict:
    if not settings.eval_enabled:
        return {}

    scores = {}

    # Faithfulness — does the answer stay grounded in the context
    if context:
        faithfulness_prompt = f"""Rate how faithful the answer is to the provided context.
Context: {context}
Question: {query}
Answer: {response}

Score from 0.0 to 1.0 where:
1.0 = completely grounded in context, no hallucinations
0.0 = answer contradicts or ignores context entirely

Respond with ONLY a number."""

        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        result = client.chat.completions.create(
            model=settings.eval_model,
            messages=[{"role": "user", "content": faithfulness_prompt}],
            max_tokens=5,
        )
        try:
            faithfulness_score = float(result.choices[0].message.content.strip())
        except ValueError:
            faithfulness_score = 0.5

        langfuse.score(
            trace_id=trace_id,
            name="faithfulness",
            value=faithfulness_score,
            comment="LLM-as-judge faithfulness score",
        )
        scores["faithfulness"] = faithfulness_score

    # Relevance — does the answer address the question
    relevance_prompt = f"""Rate how relevant the answer is to the question.
Question: {query}
Answer: {response}

Score from 0.0 to 1.0 where:
1.0 = directly and completely answers the question
0.0 = completely irrelevant

Respond with ONLY a number."""

    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    result = client.chat.completions.create(
        model=settings.eval_model,
        messages=[{"role": "user", "content": relevance_prompt}],
        max_tokens=5,
    )
    try:
        relevance_score = float(result.choices[0].message.content.strip())
    except ValueError:
        relevance_score = 0.5

    langfuse.score(
        trace_id=trace_id,
        name="relevance",
        value=relevance_score,
        comment="LLM-as-judge relevance score",
    )
    scores["relevance"] = relevance_score

    return scores
