import os
import wikipedia
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Flask app set up ---
app = Flask(__name__)
CORS(app, origins=["https://reemdelziz.github.io"])  # Allow frontend access

# --- Article list ---
related_articles = [
    "United States",
    "History of the United States",
    "Economy of the United States",
    "U.S. Constitution",
    "Politics of the United States",
    "Foreign relations of the United States",
    "Demographics of the United States",
    "Culture of the United States",
    "Education in the United States",
    "List of presidents of the United States",
]

# --- Helper to extract Wikipedia tables ---
def extract_wikipedia_tables(title: str, max_tables=3):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch HTML for {title}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')
    print(f"📊 Found {len(tables)} tables in {title}")

    documents = []
    for i, table in enumerate(tables[:max_tables]):
        rows = table.find_all('tr')
        headers = [th.get_text(strip=True) for th in rows[0].find_all(['th', 'td'])]

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if len(cells) != len(headers):
                continue  # Skip malformed
            fact = ", ".join(f"{k}: {v}" for k, v in zip(headers, cells))
            documents.append(Document(page_content=fact, metadata={"source": title, "type": "table"}))
    return documents

# --- Fetch articles and create documents ---
print("📥 Downloading Wikipedia articles...")
all_documents = []
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for title in related_articles:
    try:
        page = wikipedia.page(title)
        # Chunk the main article content
        chunks = splitter.create_documents([page.content])
        for doc in chunks:
            doc.metadata["source"] = title
        all_documents.extend(chunks)

        # Parse tables
        table_docs = extract_wikipedia_tables(title)
        all_documents.extend(table_docs)

        print(f"✔️ Processed: {title} — {len(chunks)} text chunks, {len(table_docs)} tables")
    except Exception as e:
        print(f"❌ Failed to process {title}: {e}")

print(f"\n📄 Total documents: {len(all_documents)}")

# --- Embed & Create Vector Store ---
print("🔍 Embedding with OpenAI...")
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(all_documents, embeddings)

# --- Load OpenAI Chat Model ---
print("🧠 Loading ChatGPT (gpt-3.5-turbo)...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- Build RAG Chain ---
retriever = db.as_retriever(search_type="similarity", k=4)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# --- API: Ask a question ---
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('question', '').strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    query_vector = embeddings.embed_query(query)
    results_with_scores = db.similarity_search_with_score_by_vector(query_vector, k=4)
    similarities = [1 / (1 + score) for _, score in results_with_scores]
    avg_similarity = float(sum(similarities) / len(similarities))

    result = qa.invoke({"query": query})
    raw_answer = result["result"]
    source_docs = result.get("source_documents", [])

    print(result)
    print(source_docs)

    # Format sources: article title + chunk
    sources = []
    for doc in source_docs:
        title = doc.metadata.get("source", "Unknown")
        chunk = doc.page_content.strip()
        sources.append({
            "title": title,
            "content": chunk
        })

    print(sources)

    CONFIDENCE_THRESHOLD = 0.5
    if avg_similarity < CONFIDENCE_THRESHOLD:
        raw_answer = "🤔 I'm not confident in the answer. Please rephrase the question or submit documentation I may use."

    return jsonify({
        "answer": raw_answer.strip(),
        "confidence": round(avg_similarity, 3),
        "sources": sources
    })

# --- API: Submit a new fact ---
@app.route('/submit', methods=['POST'])
def submit_fact():
    data = request.get_json()
    fact = data.get("fact", "").strip()
    source = data.get("source", "User Submission").strip()

    if not fact:
        return jsonify({"error": "No fact provided"}), 400

    new_doc = Document(page_content=fact, metadata={"source": source})
    db.add_documents([new_doc])
    print(f"➕ New fact added: {fact[:60]}...")
    return jsonify({"message": "Fact added successfully"})

# --- Run the server ---
if __name__ == '__main__':
    app.run(debug=True)
