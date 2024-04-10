import os
import torch
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    if embedding_model_name == "text-embedding-3-large":
        embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return embedding_model


class EmbedChunks:
    def __init__(self, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"device": device, "batch_size": 100},
        )

    def __call__(self, batch):
        texts = [chunk["text"] for chunk in batch]
        embeddings = self.embedding_model.embed_documents(texts)
        for i, chunk in enumerate(batch):
            chunk["embeddings"] = embeddings[i]
        return batch
    
    def process_chunks(self, chunks, batch_size=100):
        embedded_chunks = []
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            embedded_batch = self(batch) 
            embedded_chunks.extend(embedded_batch)
        return embedded_chunks
    

# embedding_model_name = "thenlper/gte-base"
# embedder = EmbedChunks(model_name=embedding_model_name)

# # Processing chunks in batches
# batch_size = 100
# embedded_chunks = []
# for i in tqdm(range(0, len(chunks), batch_size)):
#     batch = chunks[i:i+batch_size]
#     embedded_batch = embedder(batch)
#     embedded_chunks.extend(embedded_batch)