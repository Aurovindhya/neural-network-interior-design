from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.auth import authenticate_user, create_access_token, decode_token
from src.agent import create_agent, run_agent
from src.config import get_settings
import uuid

settings = get_settings()
app = FastAPI(title="HealthAgent MCP", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# In-memory session store — use Redis in production
SESSIONS: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    role: str
    eval_scores: dict


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    username = payload.get("sub")
    from src.auth import MOCK_USERS
    user = MOCK_USERS.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")

    token = create_access_token(data={"sub": user["username"], "role": user["role"]})
    session_id = str(uuid.uuid4())

    SESSIONS[session_id] = {
        "user": user,
        "agent": create_agent(user, session_id),
    }

    return {
        "access_token": token,
        "token_type": "bearer",
        "session_id": session_id,
        "role": user["role"],
        "name": user["name"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    session = SESSIONS.get(request.session_id)
    if not session:
        session_id = request.session_id
        SESSIONS[session_id] = {
            "user": current_user,
            "agent": create_agent(current_user, session_id),
        }
        session = SESSIONS[session_id]

    trace_id = str(uuid.uuid4())
    result = run_agent(session["agent"], request.message, trace_id)

    return ChatResponse(
        response=result["response"],
        session_id=request.session_id,
        role=current_user["role"],
        eval_scores=result["eval_scores"],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "HealthAgent MCP"}


@app.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {
        "name": current_user["name"],
        "role": current_user["role"],
    }
