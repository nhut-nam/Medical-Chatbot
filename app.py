from flask import Flask, render_template, jsonify, request
from src.helpers import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from store_index import docsearch
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOllama(model="mistral", temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form.get('msg')
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)