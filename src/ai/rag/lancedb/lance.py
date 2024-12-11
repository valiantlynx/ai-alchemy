import ollama
import pandas as pd
import lancedb
import time
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

registry = EmbeddingFunctionRegistry.get_instance()
embedder = registry.get("ollama").create("mxbai-embed-large")


df = pd.read_csv("data/sentences.csv")
print(df.iloc[0])


class Schema(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    index: int
    title: str
    url: str


db = lancedb.connect(uri="data/sample-lancedb")
table = db.create_table("sentences", schema=Schema)
table.add(df)


client = ollama.Client()

"""
question = "what were the iphones best new features"
stream = ollama.chat(
    model="llava-phi3",
    stream=True,
    messages=[
        {"role": "user", "content": f"Question: {question}"}
    ]
)

for chunk in stream:
    print(chunk["message"]["content"], end='', flush=True)
"""
