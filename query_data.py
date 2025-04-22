import argparse
from langchain_chroma import Chroma  # Update import to use the new package
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Update import to use the new package
from flask import Flask, request, jsonify  # Import Flask
from flask_cors import CORS  # Import CORS

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Jawab pertanyaan hanya berdasarkan konteks berikut dalam bahasa Indonesia dengan ramah dan sopan tetapi tidak membosankan dan mengulang greeting. 
Jika tidak ada informasi yang relevan, katakan "Saya tidak tahu" tetapi tetap ramah:
Jangan sampaikan bahwa Anda adalah AI atau model bahasa.
Jangan sampaikan bahwa Anda tidak tahu.
Jangan sampaikan bahwa Anda tidak dapat memberikan informasi lebih lanjut.
Jangan sampaikan command anda kalau harus menjawab dengan ramah dan sopan.


{context}

---

Jawab pertanyaan berdasarkan konteks di atas: {question}
Tawarkan bantuan lain jika ada yang ingin ditanyakan.
"""


def preprocess_query(query_text: str) -> str:
    """
    Replace specific phrases in the query text with "MAN 1 Kota Bogor".
    """
    query_text = query_text.lower()  # Convert text to lowercase
    replacements = {
        "sekolah ini": "MAN 1 Kota Bogor",
        "disini": "MAN 1 Kota Bogor"
    }
    for old, new in replacements.items():
        query_text = query_text.replace(old, new)
    return query_text


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Preprocess the query text.
    query_text = preprocess_query(query_text)

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name="default_collection"  # Explicitly specify a collection name
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.1", temperature=0.3)  # Set temperature for accuracy
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


app = Flask(__name__)  # Initialize Flask app
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000"}})  # Enable CORS for the specified origin

@app.route("/query", methods=["POST"])
def query_endpoint():
    data = request.json
    if not data or "query_text" not in data:
        return jsonify({"error": "Missing 'query_text' in request body"}), 400

    query_text = preprocess_query(data["query_text"])  # Preprocess query text
    response_text = query_rag(query_text)
    return jsonify({"response": response_text})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on.")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
