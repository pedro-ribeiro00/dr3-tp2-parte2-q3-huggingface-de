from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.runnables import RunnableLambda

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
modelo = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
modelo.eval()


class Entrada(BaseModel):
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def validar_texto(cls, texto):
        if not texto.strip():
            raise ValueError("Informe um texto.")
        return texto

def traduzir_texto(texto):
    frases = re.split(r"(?<=[.!?])\s+|\n+", texto.strip())
    blocos = []
    for frase in frases:
        tokens = tokenizer.encode(frase, add_special_tokens=False)
        for inicio in range(0, len(tokens), 350):
            blocos.append(tokens[inicio:inicio + 350])

    traducoes = []
    with torch.inference_mode():
        for bloco in blocos:
            dados = tokenizer.prepare_for_model(bloco)
            dados = {nome: torch.tensor([valor]) for nome, valor in dados.items()}
            saida = modelo.generate(**dados, max_new_tokens=510)
            if saida[0, -1].item() != tokenizer.eos_token_id:
                raise HTTPException(422, "Envie um trecho menor para traduzir.")
            traducoes.append(tokenizer.decode(saida[0], skip_special_tokens=True))
    return " ".join(traducoes)

cadeia = RunnableLambda(traduzir_texto)

@app.post("/translate")
def traduzir(entrada: Entrada):
    traducao = cadeia.invoke(entrada.text)
    return {"translation": traducao}
