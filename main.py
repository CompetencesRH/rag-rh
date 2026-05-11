# =============================================================================
# RAG PIPELINE — Accord de Télétravail
# Modèle LLM : Cerebras (llama-3.1-8b) | Embeddings : HuggingFace (local)
# Vectorstore : ChromaDB (en mémoire)
# =============================================================================
# INSTALLATION (à exécuter en première cellule Colab) :
#
# !pip install -q \
#   langchain langchain-text-splitters langchain-community \
#   langchain-huggingface langchain-chroma \
#   sentence-transformers cerebras-cloud-sdk chromadb
#!pip install cerebras-cloud-sdk
#!pip install -q \
  langchain \
  langchain-text-splitters \
  langchain-community \
  langchain-huggingface \
  langchain-chroma \
  sentence-transformers \
  cerebras-cloud-sdk \
  chromadb
#
#
# Puis : Runtime → Restart session → exécuter les cellules dans l'ordre
# =============================================================================

import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.schema import Document
from langchain_core.documents import Document
from cerebras.cloud.sdk import Cerebras




# =============================================================================
# CONFIGURATION
# =============================================================================

# Chemin du fichier Markdown dans Google Colab
# Pour vérifier le nom exact : import os; os.listdir("/content/")
#FILE_PATH = "/content/votre_document.md"

# Clé API Cerebras — à remplacer par votre clé réelle
# Obtenez-en une sur : https://cloud.cerebras.ai
#CEREBRAS_API_KEY = "votre_cle_ici"
FILE_PATH = "RefRGPD.md"
CEREBRAS_API_KEY = "csk-dj83jkyh6xxm5c5jhrpx5vhfrfhve4c46kvye26ywfre4ncn"

# =============================================================================
# ÉTAPE 1 — CHARGEMENT DU DOCUMENT
# =============================================================================

print("📄 Chargement du document...")

with open(FILE_PATH, "r", encoding="utf-8") as f:
    markdown_content = f.read()

print(f"✅ Document chargé ({len(markdown_content)} caractères)")


# =============================================================================
# ÉTAPE 2 — EXTRACTION DES TABLEAUX (Stratégie 13)
#
# Problème : les tableaux Markdown se font "mutiler" par les splitters classiques.
# Solution : on les extrait AVANT le chunking et on les traite séparément.
# Les tableaux seront réinsérés dans le vectorstore comme chunks indépendants.
# =============================================================================

def extract_tables(text):
    """
    Détecte et extrait les tableaux Markdown du texte.
    Retourne le texte nettoyé + la liste des tableaux trouvés.

    Format détecté :
        | Col1 | Col2 |
        |------|------|
        | val1 | val2 |
    """
    # Regex : ligne avec pipes | ... | suivie d'une ligne de séparation |---|
    table_pattern = r'(\|.+\|[\n\r](?:\|[-:| ]+\|[\n\r])(?:\|.+\|[\n\r])*)'
    tables = re.findall(table_pattern, text)

    # On supprime les tableaux du texte principal pour éviter les doublons
    clean_text = re.sub(table_pattern, '\n', text)

    print(f"📊 {len(tables)} tableau(x) extrait(s) séparément")
    return clean_text, tables


markdown_clean, tables = extract_tables(markdown_content)


# =============================================================================
# ÉTAPE 3 — CHUNKING HYBRIDE (Stratégies 7 + 17 + 3)
#
# Stratégie 7  — Structured Chunking : on découpe d'abord par titres Markdown
#                (#, ##, ###) pour respecter la structure du document.
#
# Stratégie 17 — Recursive Chunking : les sections trop longues sont ensuite
#                redécoupées récursivement (paragraphe → phrase → mot).
#
# Stratégie 3  — Sliding Window : un overlap de 200 caractères (~20%) évite
#                de perdre le contexte aux jonctions entre chunks.
# =============================================================================

print("\n✂️  Découpage du document en chunks...")

# --- 3a. Découpage par titres Markdown (Stratégie 7) ---
headers_to_split_on = [
    ("#",   "Header 1"),   # Titre principal
    ("##",  "Header 2"),   # Sous-section
    ("###", "Header 3"),   # Sous-sous-section
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False   # On garde les titres dans le contenu des chunks
)
md_header_splits = markdown_splitter.split_text(markdown_clean)

# --- 3b. Redécoupage des sections trop longues (Stratégie 17 + 3) ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Taille max d'un chunk (en caractères)
    chunk_overlap=200,     # Overlap de 20% pour ne pas couper le contexte
    separators=[           # Ordre de priorité : du plus gros au plus petit
        "\n\n",            # Paragraphes
        "\n",              # Sauts de ligne simples
        ".",               # Phrases
        " ",               # Mots
        ""                 # Caractères (dernier recours)
    ]
)
splits = text_splitter.split_documents(md_header_splits)

print(f"✅ {len(splits)} chunks créés depuis le texte principal")


# =============================================================================
# ÉTAPE 4 — FIX : RÉINJECTION DU TITRE DANS CHAQUE CHUNK (Stratégie 15)
#
# Problème : MarkdownHeaderTextSplitter met les titres dans les MÉTADONNÉES
# mais PAS dans le texte du chunk → le modèle d'embeddings ne les "voit" pas
# → impossible de retrouver "Accord de Télétravail" par similarité.
#
# Solution : on préfixe chaque chunk avec sa hiérarchie de titres complète.
# Ex : "[Accord de Télétravail > Article 3 > Conditions]\n\n<contenu>"
# =============================================================================

def enrich_chunks_with_context(splits):
    """
    Préfixe chaque chunk avec sa hiérarchie de titres extraite des métadonnées.
    Cela rend le titre visible pour les embeddings ET améliore la retrieval.
    """
    for doc in splits:
        # Reconstruction du chemin hiérarchique depuis les métadonnées
        prefix_parts = []
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in doc.metadata:
                prefix_parts.append(doc.metadata[key])

        if prefix_parts:
            prefix = " > ".join(prefix_parts)
            # On injecte le contexte en tête du chunk
            doc.page_content = f"[{prefix}]\n\n{doc.page_content}"

    return splits


splits = enrich_chunks_with_context(splits)


# =============================================================================
# ÉTAPE 5 — FIX : CHUNK DÉDIÉ AU TITRE PRINCIPAL
#
# Le titre principal "# Accord de Télétravail" est souvent ignoré car il
# constitue à lui seul un "chunk vide" après le découpage par headers.
# On crée un chunk explicite pour qu'il soit toujours retrouvable.
# =============================================================================

def add_title_chunk(markdown_text, splits):
    """
    Crée un chunk dédié pour le titre principal (ligne commençant par '# ').
    Garantit que "Quel est le titre de ce document ?" trouve une réponse.
    """
    lines = markdown_text.strip().split("\n")
    # On cherche la première ligne qui commence par '# ' (titre niveau 1)
    title_line = next((l for l in lines if l.startswith("# ")), None)

    if title_line:
        title = title_line.replace("# ", "").strip()
        print(f"📌 Titre principal détecté : '{title}'")

        title_doc = Document(
            # Formulations multiples pour maximiser la chance de retrieval
            page_content=(
                f"Ce document est intitulé : {title}\n"
                f"Titre principal du document : {title}\n"
                f"Il s'agit d'un accord portant sur : {title}"
            ),
            metadata={"Header 1": title, "source": "titre_principal"}
        )
        # On l'insère EN PREMIER pour lui donner la priorité
        splits.insert(0, title_doc)
    else:
        print("⚠️  Aucun titre principal (# ...) trouvé dans le document")

    return splits


splits = add_title_chunk(markdown_content, splits)


# =============================================================================
# ÉTAPE 6 — AJOUT DES TABLEAUX COMME CHUNKS INDÉPENDANTS (Stratégie 13)
#
# Chaque tableau extrait à l'étape 2 devient un chunk séparé,
# préservant ainsi sa structure tabulaire pour la retrieval.
# =============================================================================

for i, table in enumerate(tables):
    table_doc = Document(
        page_content=f"[Tableau {i+1}]\n\n{table.strip()}",
        metadata={"source": f"tableau_{i+1}"}
    )
    splits.append(table_doc)

print(f"✅ {len(splits)} chunks au total (texte + titre + tableaux)")


# =============================================================================
# ÉTAPE 7 — EMBEDDINGS ET VECTORSTORE
#
# On utilise un modèle d'embeddings local (gratuit, pas de clé API nécessaire).
# "all-MiniLM-L6-v2" est rapide et performant pour le français et l'anglais.
# ChromaDB stocke les vecteurs en mémoire (pas de fichier persistant ici).
# =============================================================================

print("\n🔢 Création des embeddings et du vectorstore...")
print("   (première exécution : téléchargement du modèle ~90Mo, patientez...)")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

print(f"✅ Vectorstore créé avec {len(splits)} chunks indexés")


# =============================================================================
# ÉTAPE 8 — CLIENT CEREBRAS (LLM pour la génération)
# =============================================================================

client = Cerebras(api_key=CEREBRAS_API_KEY)


# =============================================================================
# ÉTAPE 9 — FONCTION RAG PRINCIPALE
# =============================================================================

def ask_rag(question: str, k: int = 3, verbose: bool = False) -> str:
    """
    Pipeline RAG complet : Retrieval → Augmentation → Generation

    Args:
        question : La question posée par l'utilisateur
        k        : Nombre de chunks à récupérer (défaut : 3)
        verbose  : Si True, affiche les chunks récupérés (utile pour debug)

    Returns:
        La réponse générée par le LLM Cerebras
    """

    # --- RETRIEVAL : Recherche MMR (Maximal Marginal Relevance) ---
    # Avantage sur la similarité cosine classique :
    # MMR équilibre pertinence ET diversité → évite 3 chunks quasi identiques
    # fetch_k=10 : on récupère 10 candidats, puis on garde les k plus diversifiés
    # lambda_mult : 0 = max diversité | 1 = max similarité (0.7 = bon équilibre)
    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=10,
        lambda_mult=0.7
    )

    # Affichage debug optionnel
    if verbose:
        print(f"\n🔍 {len(docs)} chunks récupérés pour : '{question}'")
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:300])

    # --- AUGMENTATION : Construction du contexte ---
    context = "\n\n".join([doc.page_content for doc in docs])

    # --- PROMPT : Instructions claires pour le LLM ---
    prompt = f"""Tu es un assistant RH expert en droit du travail français.
Utilise UNIQUEMENT le CONTEXTE fourni pour répondre à la QUESTION.
Si la réponse n'est pas dans le contexte, dis poliment que tu ne trouves pas cette information dans le document.
Ne jamais inventer d'information absente du contexte.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

    # --- GENERATION : Appel à l'API Cerebras ---
    response = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,      # Limite la longueur de la réponse
        temperature=0.1,     # Faible température = réponses factuelles, peu créatives
    )

    return response.choices[0].message.content


# =============================================================================
# ÉTAPE 10 — TESTS
# =============================================================================

print("\n" + "="*60)
print("🚀 TESTS DU PIPELINE RAG")
print("="*60)

questions_test = [
    "Quel est le titre de ce document ?",
    "De quoi parle ce document ?",
    "Quelles sont les principes fondamentaux du traitement ?",
    "Dans le cadre de l'ia Act 2026, que dire de la transparence et interpretabilité ?",
    "Combien de délai de conservation et archivage pour un dossier candidat ?",
]

for question in questions_test:
    print(f"\n❓ {question}")
    print(f"💬 {ask_rag(question)}")
    print("-" * 60)


# =============================================================================
# UTILISATION INTERACTIVE
# =============================================================================

# Pour poser vos propres questions, utilisez directement :
#
# reponse = ask_rag("Votre question ici")
# print(reponse)
#
# Pour voir les chunks utilisés (mode debug) :
# reponse = ask_rag("Votre question", verbose=True)
# print(reponse)
