import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingest import ingest_document
from config import Config

load_dotenv()

app = Flask(__name__)

# ---------------- Models ----------------

embeddings = None
vectordb = None
llm = None

def load_models():
    global embeddings, vectordb, llm

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

    if vectordb is None:
        vectordb = Chroma(
            persist_directory=Config.VECTOR_DB_PATH,
            embedding_function=embeddings
        )

    if llm is None:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

# Load models when server starts
load_models()

# ---------------- Home ----------------

@app.route("/")
def home():
    return "AI RAG Service Running"

# ---------------- Health ----------------

@app.route("/health")
def health():
    return "AI Service OK"

# ---------------- Upload Document ----------------

@app.route("/upload", methods=["POST"])
def upload_document():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    save_path = os.path.join(Config.DATA_PATH, file.filename)

    file.save(save_path)

    ingest_document(save_path)

    return jsonify({
        "message": "Document uploaded and indexed"
    })

# ---------------- Ask Question ----------------

@app.route("/ask", methods=["POST"])
def ask_ai():

    data = request.get_json() or {}
    question = data.get("question","").strip()

    if not question:
        return jsonify({"error":"Question required"}),400

    docs = vectordb.similarity_search_with_score(question, k=3)

    docs = [d for d in docs if d[1] < 2]

    if not docs:

        answer = llm.invoke([
            SystemMessage(content="Answer briefly using general knowledge"),
            HumanMessage(content=question)
        ]).content

        return jsonify({
            "answer": answer,
            "source": "general knowledge"
        })

    context = "\n".join(d[0].page_content[:1000] for d in docs)

    answer = llm.invoke([
        SystemMessage(content="""
You are a document assistant.
Answer only from the given context.
If answer not found say 'Not found in document'.
Keep answer short.
"""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:{question}")
    ]).content

    sources = list(set(d[0].metadata.get("source","Unknown") for d in docs))

    return jsonify({
        "answer": answer,
        "sources": sources
    })

# ---------------- Run ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)