from src.chatbot import Chatbot
from src.data_parse import read_pdf
from fastapi import FastAPI
from fastapi import UploadFile
import uvicorn
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

chatBot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatBot
    chatBot = Chatbot()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/add_pdf")
async def add_pdf(file: UploadFile):
    try:
        read_pdf(file=file)
    except Exception as e:
        return JSONResponse(
            content={"error": f"{e}"}, status_code=500
        )
    return JSONResponse(
        content={"status": f"successfully added {file.filename}"},
        status_code=200,
    )


@app.get("/chat_here")
async def chat_here(question: str):
    try:
        answer = chatBot.chat(question)
    except Exception as e:
        return JSONResponse(
            content={"error": f"{e}"}, status_code=500
        )
    return JSONResponse(
        content={"answer": answer}, status_code=200
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
