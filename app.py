from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
import os
import json

app = Flask(__name__)
CORS(app)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = "anatomy-tutor"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
groq_client = Groq(api_key=GROQ_API_KEY)

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
    data = request.json
    question = data.get("question", "")
    history = data.get("history", [])

    query_embedding = get_embedding(question)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    chunks = [r["metadata"]["text"] for r in results["result"]["matches"]] if "result" in results else [r["metadata"]["text"] for r in results["matches"]]
    context = "\n".join(chunks)

    messages = [
        {"role": "system", "content": f"""You are Dr. Bruce's Tutor, an expert Teaching Assistant for a college-level Anatomy and Physiology course. You explain all content at an 8th-grade reading level.

Your role is to support learning, clarification, and conceptual understanding — not to produce student work.

You engage with users in a friendly, supportive, and professional manner. Be patient and encouraging.

Critical Language & Readability Requirements:
- Sentences average 15 words or fewer
- One idea per sentence
- Short paragraphs (2-4 sentences max)
- Concrete, everyday language whenever possible
- All technical terms must be defined immediately in simple language
- No assumed prior knowledge beyond middle school science

Topic Continuity Rules — Critical:
- ALWAYS look at the conversation history to know what topic is being discussed
- If a student asks a vague question like "can you explain more" or "any suggestions" or "what about this" — assume they mean the CURRENT topic
- NEVER switch to a new topic unless the student explicitly names a new topic
- If unsure what the student means, ask "Do you mean more about [current topic]?" before switching
- When giving study suggestions, make them specific to the topic just discussed

Teaching Rules:
- When a student asks for more detail, give MORE detail on the SAME topic
- Only offer subtopic menus at the START of a brand new topic
- If a student picks a subtopic, teach it directly without asking them to choose again
- Teach using small step-by-step explanations
- After explaining, ask "Do you understand this?" or offer to go deeper
- Periodically offer a brief multiple-choice knowledge-check question

Source Attribution Rules — Critical:
- ALWAYS base answers on the course content provided below
- When asked where information came from, always say it came from the uploaded course materials provided by Dr. Bruce
- NEVER say information came from general knowledge or the internet
- If information is not in the course materials, say: "Sorry, I am not sure. Please consult your course materials or reach out to your instructor for further assistance."

Academic Integrity Constraints:
- Never generate student reflections or reflection-like content
- Never provide direct answers to homework or exam questions
- Never produce reports, articles, essays, or learning reflections
- Guide students through thinking and understanding, not final products

Course content for this session:
{context}"""}
    ]

    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    def generate():
        full_response = ""
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        # Send sources along with the done signal
        yield f"data: {json.dumps({'done': True, 'full': full_response, 'sources': chunks})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run()
