import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# =============================================================================
# CONFIGURATION
# =============================================================================
DOCS_FOLDER = "documents"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CORS (même config que le chatbot)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://competencesrh.fr", "https://www.competencesrh.fr"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Variables globales
vectorstore = None
current_doc_name = None

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # quota gratuit le plus généreux (14 400 req/jour)
    temperature=0.1,
    api_key=GROQ_API_KEY
)

SYSTEM_PROMPT_TEMPLATE = """Tu es l'assistant RH expert de CompétencesRH, spécialisé en droit du travail français.
Utilise UNIQUEMENT le CONTEXTE fourni ci-dessous pour répondre à la QUESTION.
Si la réponse n'est pas dans le contexte, dis poliment que tu ne trouves pas l'info et recommande de contacter le service RH.
Ne jamais inventer une règle légale précise.
Langue : français uniquement.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

# =============================================================================
# MODELES
# =============================================================================
class SelectDocRequest(BaseModel):
    filename: str

class AskRequest(BaseModel):
    question: str

# =============================================================================
# FONCTION D'INITIALISATION DYNAMIQUE
# =============================================================================
def load_document_to_rag(filename: str):
    global vectorstore, current_doc_name

    file_path = os.path.join(DOCS_FOLDER, filename)
    if not os.path.exists(file_path):
        return False, f"Fichier {filename} introuvable."

    print(f"📄 Chargement de {filename}...")
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # 1. Extraction des tableaux
    def extract_tables(text):
        table_pattern = r'(\|.+\|[\n\r](?:\|[-:| ]+\|[\n\r])(?:\|.+\|[\n\r])*)'
        tables = re.findall(table_pattern, text)
        clean_text = re.sub(table_pattern, '\n', text)
        return clean_text, tables

    markdown_clean, tables = extract_tables(markdown_content)

    # 2. Chunking
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(markdown_clean)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(md_header_splits)

    # 3. Enrichissement contexte
    for doc in splits:
        prefix_parts = [doc.metadata[k] for k in ["Header 1", "Header 2", "Header 3"] if k in doc.metadata]
        if prefix_parts:
            doc.page_content = f"[{' > '.join(prefix_parts)}]\n\n{doc.page_content}"

    # 4. Tableaux
    for i, table in enumerate(tables):
        splits.append(Document(
            page_content=f"[Tableau {i+1}]\n\n{table.strip()}",
            metadata={"source": f"tableau_{i+1}"}
        ))

    # 5. Embeddings et Vectorstore (on écrase l'ancien pour libérer la RAM)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN
    )

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    current_doc_name = filename
    print(f"✅ {filename} indexé avec succès.")
    return True, f"{filename} chargé avec succès."

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
async def index(request: Request):
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".md")]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "files": files, "current_doc": current_doc_name}
    )

@app.post("/select_doc")
async def select_doc(req: SelectDocRequest):
    if not req.filename.strip():
        raise HTTPException(status_code=400, detail="Aucun fichier sélectionné")

    success, message = load_document_to_rag(req.filename)
    if success:
        return {"message": message}
    raise HTTPException(status_code=500, detail=message)

@app.post("/ask")
async def ask(req: AskRequest):
    if vectorstore is None:
        return {"answer": "Veuillez d'abord sélectionner un document dans la liste."}

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        # Recherche des chunks pertinents
        docs = vectorstore.max_marginal_relevance_search(req.question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=req.question)

        result = llm.invoke([("human", prompt)])
        return {"answer": result.content}
    except Exception as e:
        return {"answer": f"Erreur : {str(e)[:200]}"}

# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
