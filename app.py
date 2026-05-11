# app.py — Ton RAG RGPD exact (Flask + Render)
import os
from flask import Flask, render_template, request, jsonify

# =============================================================================
# TON CODE EXACT (importé et gardé à 100%)
# =============================================================================
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

# Configuration (TA clé via Render)
FILE_PATH = "RefRGPD.md"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")  # ← Via Render !

# Variables globales pour ton vectorstore et client
vectorstore = None
client = None

# =============================================================================
# FONCTION D'INITIALISATION (charge ton code au démarrage)
# =============================================================================
def init_rag():
    global vectorstore, client
    
    print("📄 Chargement du document...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    print(f"✅ Document chargé ({len(markdown_content)} caractères)")

    # TON ÉTAPE 2 — EXTRACTION DES TABLEAUX (exactement ton code)
    def extract_tables(text):
        table_pattern = r'(\|.+\|[\n\r](?:\|[-:| ]+\|[\n\r])(?:\|.+\|[\n\r])*)'  # ← Corrigé
        tables = re.findall(table_pattern, text)
        clean_text = re.sub(table_pattern, '\n', text)  # ← Corrigé
        print(f"📊 {len(tables)} tableau(x) extrait(s) séparément")
        return clean_text, tables

    markdown_clean, tables = extract_tables(markdown_content)

    # TON ÉTAPE 3 — CHUNKING HYBRIDE (exactement ton code)
    print("\n✂️  Découpage du document en chunks...")
    headers_to_split_on = [
        ("#",   "Header 1"),   ("##",  "Header 2"),   ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_clean)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(md_header_splits)
    print(f"✅ {len(splits)} chunks créés depuis le texte principal")

    # TON ÉTAPE 4 — RÉINJECTION TITRE (exactement)
    def enrich_chunks_with_context(splits):
        for doc in splits:
            prefix_parts = []
            for key in ["Header 1", "Header 2", "Header 3"]:
                if key in doc.metadata:
                    prefix_parts.append(doc.metadata[key])
            if prefix_parts:
                prefix = " > ".join(prefix_parts)
                doc.page_content = f"[{prefix}]\n\n{doc.page_content}"
        return splits
    splits = enrich_chunks_with_context(splits)

    # TON ÉTAPE 5 — CHUNK TITRE PRINCIPAL (exactement)
    def add_title_chunk(markdown_text, splits):
        lines = markdown_text.strip().split("\n")
        title_line = next((l for l in lines if l.startswith("# ")), None)
        if title_line:
            title = title_line.replace("# ", "").strip()
            print(f"📌 Titre principal détecté : '{title}'")
            title_doc = Document(
                page_content=f"Ce document est intitulé : {title}\nTitre principal du document : {title}\nIl s'agit d'un accord portant sur : {title}",
                metadata={"Header 1": title, "source": "titre_principal"}
            )
            splits.insert(0, title_doc)
        else:
            print("⚠️  Aucun titre principal (# ...) trouvé dans le document")
        return splits
    splits = add_title_chunk(markdown_content, splits)

    # TON ÉTAPE 6 — TABLEAUX (exactement)
    for i, table in enumerate(tables):
        table_doc = Document(
            page_content=f"[Tableau {i+1}]\n\n{table.strip()}",
            metadata={"source": f"tableau_{i+1}"}
        )
        splits.append(table_doc)
    print(f"✅ {len(splits)} chunks au total (texte + titre + tableaux)")

    # TON ÉTAPE 7 — EMBEDDINGS (exactement)
    print("\n🔢 Création des embeddings et du vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print(f"✅ Vectorstore créé avec {len(splits)} chunks indexés")

    # TON ÉTAPE 8 — CLIENT CEREBRAS
    client = Cerebras(api_key=CEREBRAS_API_KEY)

# =============================================================================
# TA FONCTION RAG PRINCIPALE (exactement conservée)
# =============================================================================
def ask_rag(question: str, k: int = 3, verbose: bool = False) -> str:
    if not vectorstore or not client:
        return "❌ RAG non initialisé. Vérifiez la clé API et RefRGPD.md"
    
    docs = vectorstore.max_marginal_relevance_search(question, k=k, fetch_k=10, lambda_mult=0.7)
    if verbose:
        print(f"\n🔍 {len(docs)} chunks récupérés pour : '{question}'")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300])
    
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Tu es un assistant RH expert en droit du travail français.
Utilise UNIQUEMENT le CONTEXTE fourni pour répondre à la QUESTION.
Si la réponse n'est pas dans le contexte, dis poliment que tu ne trouves pas cette information dans le document.
Ne jamais inventer d'information absente du contexte.

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
# ROUTES WEB (page vitrine + API)
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    if question:
        answer = ask_rag(question)
        return jsonify({"answer": answer})
    return jsonify({"error": "Aucune question"})

if __name__ == "__main__":
    init_rag()  # ← Charge ton RAG au démarrage
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
