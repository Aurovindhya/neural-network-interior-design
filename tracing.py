"""
tracing.py

Langfuse integration for tracing inference calls.
Degrades gracefully if Langfuse keys are not configured.

Usage:
    from langfuse.tracing import get_tracer

    tracer = get_tracer()
    with tracer.trace("predict") as span:
        span.set_input({"filename": "room.jpg", "size": [800, 600]})
        result = model.predict(img)
        span.set_output(result)
"""

import os
import time
import uuid
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class _Span:
    """Lightweight span object returned by tracer.trace()."""

    def __init__(self, trace_id: str, name: str, lf_trace=None):
        self.trace_id = trace_id
        self.name = name
        self._lf_trace = lf_trace
        self._start = time.time()
        self._input: Dict[str, Any] = {}
        self._output: Dict[str, Any] = {}

    def set_input(self, data: Dict[str, Any]):
        self._input = data
        if self._lf_trace:
            try:
                self._lf_trace.update(input=data)
            except Exception:
                pass

    def set_output(self, data: Dict[str, Any]):
        self._output = data
        if self._lf_trace:
            try:
                self._lf_trace.update(output=data)
            except Exception:
                pass

    def set_metadata(self, key: str, value: Any):
        if self._lf_trace:
            try:
                self._lf_trace.update(metadata={key: value})
            except Exception:
                pass

    def finish(self, error: Optional[str] = None):
        latency_ms = (time.time() - self._start) * 1000
        if self._lf_trace:
            try:
                update = {"metadata": {"latency_ms": round(latency_ms, 2)}}
                if error:
                    update["level"] = "ERROR"
                    update["status_message"] = error
                self._lf_trace.update(**update)
            except Exception:
                pass
        logger.debug(f"Trace {self.trace_id} | {self.name} | {latency_ms:.1f}ms")


class _NoopTracer:
    """Fallback tracer that does nothing (Langfuse not configured)."""

    @contextmanager
    def trace(self, name: str = "predict"):
        span = _Span(trace_id=str(uuid.uuid4())[:8], name=name)
        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        else:
            span.finish()


class LangfuseTracer:
    """Real tracer backed by Langfuse SDK."""

    def __init__(self, langfuse_client):
        self._lf = langfuse_client

    @contextmanager
    def trace(self, name: str = "predict"):
        trace_id = str(uuid.uuid4())
        lf_trace = self._lf.trace(id=trace_id, name=name)
        span = _Span(trace_id=trace_id[:8], name=name, lf_trace=lf_trace)
        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        else:
            span.finish()


def get_tracer():
    """
    Returns a LangfuseTracer if keys are configured, else a no-op tracer.
    Safe to call at import time.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if public_key and secret_key:
        try:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("Langfuse tracing enabled.")
            return LangfuseTracer(client)
        except Exception as e:
            logger.warning(f"Langfuse init failed, falling back to noop tracer: {e}")

    logger.info("Langfuse keys not found — tracing disabled.")
    return _NoopTracer()
