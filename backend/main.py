from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

app = FastAPI()

# CORS (프론트 연결 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)