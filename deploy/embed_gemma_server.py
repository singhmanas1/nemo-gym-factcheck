#!/usr/bin/env python3
"""OpenAI-compatible embedding server for google/embeddinggemma-300m."""

import os
from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import uvicorn

MODEL_ID = os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m")
MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", MODEL_ID)
HOST = os.environ.get("EMBED_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBED_PORT", "8002"))
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

print(f"Loading {MODEL_ID} from {MODEL_PATH} on {DEVICE} ...", flush=True)
model = SentenceTransformer(
    MODEL_PATH,
    token=os.environ.get("HF_TOKEN"),
    device=DEVICE,
)
DIM = model.get_sentence_embedding_dimension()
print(f"Ready dim={DIM} device={model.device}", flush=True)

app = FastAPI(title="embeddinggemma-300m")


class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID
    encoding_format: str = "float"


class EmbedData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbedResponse(BaseModel):
    object: str = "list"
    data: List[EmbedData]
    model: str
    usage: dict = Field(default_factory=dict)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID, "dim": DIM, "device": str(model.device)}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> EmbedResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = model.encode(
        texts,
        prompt_name="Retrieval-query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    data = [
        EmbedData(embedding=vec.tolist(), index=i)
        for i, vec in enumerate(vectors)
    ]
    n_tokens = sum(len(t.split()) for t in texts)
    return EmbedResponse(
        data=data,
        model=req.model,
        usage={"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
