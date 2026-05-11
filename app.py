import os
import re
from flask import Flask, render_template, request, jsonify

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # Version API légère
from langchain_core.documents import Document
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
FILE_PATH = "RefRGPD.md"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Variables globales
vectorstore = None
client = None

# =============================================================================
# FONCTION D'INITIALISATION
# =============================================================================
def init_rag():
    global vectorstore, client
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Erreur : Le fichier {FILE_PATH} est introuvable.")
        return

    print("📄 Chargement du document...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    print(f"✅ Document chargé ({len(markdown_content)} caractères)")

    # Extraction des tableaux
    def extract_tables(text):
        table_pattern = r'(\|.+\|[\n\r](?:\|[-:| ]+\|[\n\r])(?:\|.+\|[\n\r])*)'
        tables = re.findall(table_pattern, text)
        clean_text = re.sub(table_pattern, '\n', text)
        return clean_text, tables

    markdown_clean, tables = extract_tables(markdown_content)

    # Chunking hybride
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(markdown_clean)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(md_header_splits)

    # Enrichissement avec contexte
    def enrich_chunks_with_context(docs):
        for doc in docs:
            prefix_parts = [doc.metadata[k] for k in ["Header 1", "Header 2", "Header 3"] if k in doc.metadata]
            if prefix_parts:
                doc.page_content = f"[{' > '.join(prefix_parts)}]\n\n{doc.page_content}"
        return docs
    
    splits = enrich_chunks_with_context(splits)

    # Ajout du titre principal
    lines = markdown_content.strip().split("\n")
    title_line = next((l for l in lines if l.startswith("# ")), None)
    if title_line:
        title = title_line.replace("# ", "").strip()
        splits.insert(0, Document(
            page_content=f"Ce document est intitulé : {title}\nIl porte sur le RGPD.",
            metadata={"Header 1": title, "source": "titre_principal"}
        ))

    # Ajout des tableaux
    for i, table in enumerate(tables):
        splits.append(Document(
            page_content=f"[Tableau {i+1}]\n\n{table.strip()}",
            metadata={"source": f"tableau_{i+1}"}
        ))

    # EMBEDDINGS VIA API (Consomme très peu de RAM)
    print("🔢 Initialisation des embeddings via Hugging Face API...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN
    )
    
    # Création du vectorstore (en mémoire pour le plan gratuit)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print(f"✅ Vectorstore prêt ({len(splits)} chunks)")

    # Client Cerebras
    client = Cerebras(api_key=CEREBRAS_API_KEY)

# =============================================================================
# LOGIQUE RAG
# =============================================================================
def ask_rag(question: str, k: int = 3) -> str:
    if not vectorstore or not client:
        return "❌ Système non prêt. Vérifiez les clés API."
    
    docs = vectorstore.max_marginal_relevance_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"""Tu es un assistant RH expert en droit du travail français.
Utilise UNIQUEMENT le CONTEXTE fourni pour répondre à la QUESTION.
Si la réponse n'est pas dans le contexte, dis poliment que tu ne trouves pas l'info.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.1,
    )
    return response.choices[0].message.content

# =============================================================================
# ROUTES FLASK
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question vide"}), 400
    
    answer = ask_rag(question)
    return jsonify({"answer": answer})

# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    init_rag() 
    # Port dynamique requis par Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
