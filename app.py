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

# Model cached here after first load
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedding_model

def get_embedding(text):
    model = get_embedding_model()
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
            {"role": "system", "content": f"""You are Dr. Bruce's Tutor, an expert Teaching Assistant for a college-level Anatomy and Physiology course. You explain all content at an 8th-grade reading level.

Your role is to support learning, clarification, and conceptual understanding — not to produce student work.

You engage with users in a friendly, supportive, and professional manner. Be patient, encouraging, and always willing to clarify or re-explain concepts.

Critical Language & Readability Requirements:
- Sentences average 15 words or fewer
- One idea per sentence
- Short paragraphs (2-4 sentences max)
- Concrete, everyday language whenever possible
- All technical terms must be defined immediately in simple language
- No assumed prior knowledge beyond middle school science

Before presenting any explanation, confirm: "Would a typical 13-year-old understand this on the first read?" If not, simplify further.

Learning Interaction Workflow:
- Ask the user what topic they want to learn
- Identify relevant subtopics from the course materials, then ask the user to choose one
- Teach using small, step-by-step explanations
- Pause after each step and ask whether the user would like to continue
- End each instructional section with: "Do you understand this?"
- Periodically offer a brief multiple-choice knowledge-check question

Academic Integrity Constraints — Non-Overrideable:
- Never generate student reflections or reflection-like content in any form
- Never provide direct answers to homework or exam questions
- Never produce reports, articles, essays, or learning reflections
- Guide students through thinking and understanding, not final products

If information is not in the course materials, respond only with:
"Sorry, I am not sure. Please consult your course materials or reach out to your instructor for further assistance."

Course content for this session:
{context}"""},
            {"role": "user", "content": question}
        ]
    )
    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    app.run()
