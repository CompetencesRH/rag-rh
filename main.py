from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://competencesrh.fr", "https://www.competencesrh.fr"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── MODELE REQUEST ─────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    message: str

# ── GLOBAL ─────────────────────────────────────────────────────────────────────
qa_chain = None

# ── STARTUP ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global qa_chain

    print("📂 Chargement des documents /docs...")

    # Charge tous les .md du dossier /docs
    loader = DirectoryLoader(
        "./docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"✅ {len(documents)} document(s) chargé(s)")

    # Découpage sur la structure Markdown
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunk(s) créé(s)")

    # Embeddings locaux — gratuits, pas d'API externe
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Base vectorielle persistée sur disque
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB créé")

    # LLM Groq — gratuit
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Chain RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    print("✅ RAG prêt")

# ── ENDPOINTS ──────────────────────────────────────────────────────────────────
@app.post("/rag")
async def rag(req: QuestionRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide.")
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG non initialisé.")

    try:
        result = qa_chain({"query": req.message.strip()})

        sources = list(set([
            os.path.basename(doc.metadata.get("source", "document interne"))
            for doc in result.get("source_documents", [])
        ]))

        return {
            "response": result["result"],
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": qa_chain is not None}

# ── RUN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
