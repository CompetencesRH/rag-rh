import os
import re
from flask import Flask, render_template, request, jsonify

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from cerebras.cloud.sdk import Cerebras

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
DOCS_FOLDER = "documents"
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Variables globales
vectorstore = None
client = Cerebras(api_key=CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None
current_doc_name = None

# =============================================================================
# FONCTION D'INITIALISATION DYNAMIQUE
# =============================================================================
def load_document_to_rag(filename):
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

    # 5. Embeddings et Vectorstore (On écrase l'ancien pour libérer la RAM)
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
# ROUTES FLASK
# =============================================================================
@app.route("/")
def index():
    # Liste les fichiers .md présents dans le dossier documents/
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".md")]
    return render_template("index.html", files=files, current_doc=current_doc_name)

@app.route("/select_doc", methods=["POST"])
def select_doc():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    success, message = load_document_to_rag(filename)
    if success:
        return jsonify({"message": message})
    return jsonify({"error": message}), 500

@app.route("/ask", methods=["POST"])
def ask():
    if vectorstore is None:
        return jsonify({"answer": "Veuillez d'abord sélectionner un document dans la liste."})

    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question vide"}), 400
    
    try:
        # Recherche des chunks
        docs = vectorstore.max_marginal_relevance_search(question, k=3)
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
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.1,
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"answer": f"Erreur : {str(e)}"}), 500

# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
