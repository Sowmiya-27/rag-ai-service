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

# ---------------- Embeddings ----------------

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


# ---------------- Home ----------------

@app.route("/")
def home():

    return "AI RAG Service Running"

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

    load_models()

    data = request.get_json()
    question = data.get("question")

    docs = vectordb.similarity_search_with_score(question, k=3)

    docs = [d for d in docs if d[1] < 2]

    if not docs:

        answer = llm.invoke([
            SystemMessage(content="Answer using general knowledge"),
            HumanMessage(content=question)
        ]).content

        return jsonify({
            "answer": answer,
            "source": "general knowledge"
        })

    context = "\n".join(d[0].page_content for d in docs)

    answer = llm.invoke([
        SystemMessage(content="Answer only from context"),
        HumanMessage(content=f"{context}\nQuestion:{question}")
    ]).content

    sources = []

    for d,_ in docs:
        sources.append(d.metadata.get("source","Unknown"))

    return jsonify({
        "answer": answer,
        "sources": list(set(sources))
    })
# ---------------- Run ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)