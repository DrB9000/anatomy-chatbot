from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = "anatomy-tutor"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
groq_client = Groq(api_key=GROQ_API_KEY)

def get_embedding(text):
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return list(model.embed([text]))[0].tolist()

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question", "")
    query_embedding = get_embedding(question)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    context = "\n".join([r["metadata"]["text"] for r in results["matches"]])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a helpful anatomy and physiology tutor. Answer questions using this content:\n\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    app.run()
