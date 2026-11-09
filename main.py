from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from schemas import TextRequest, TranslationResponse
from service import Translator
from langchain_core.runnables import RunnableLambda

@asynccontextmanager
async def lifespan(app: FastAPI):
    translator = Translator("Helsinki-NLP/opus-mt-en-de")
    app.state.translate = RunnableLambda(translator.translate)
    yield

app = FastAPI(title="parte2_q3_huggingface_de", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate", response_model=TranslationResponse)
def translate(data: TextRequest):
    try:
        result = app.state.translate.invoke(data.text)
        return {"translation": result, "target_language": "de"}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, "Falha de inferência local.") from exc
