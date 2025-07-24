import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter 
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

# Gemini API setup
genai.configure(api_key=os.getenv("GOOGLE-API-KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding model config
def embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Frontend
st.header("RAG using HF Embeddings + FAISS db")
uploaded_file = st.file_uploader("Upload the Document", type=["pdf"])

# PDF Ingestion
if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)
    
    for page in pdf.pages:
        context = page.extract_text()
        if context:
            raw_text += context

    if raw_text.strip():
        # Chunking
        document = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([document])

        # Embedding + Vector DB
        texts = [chunk.page_content for chunk in chunks]
        vector_db = FAISS.from_texts(texts, embedding_model())
        retriever = vector_db.as_retriever()

        st.success("✅ Document Processed Successfully. Ask Questions Below.")
        user_input = st.text_input("Enter your Query")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analysing the Document...."):
                retrieved_doc = retriever.get_relevant_documents(user_input)
                context = "\n".join(doc.page_content for doc in retrieved_doc)
                prompt = f'''You are an expert assistant and use the context below to answer the 
                query. If Unsure, just say I don't Know...khai or jaakr dekh. 
                Context : {context},
                User Query : {user_input}
                Answer :  '''
                response = model.generate_content(prompt)
                st.markdown("Answer:")
                st.write(response.text)
        else:
            st.warning("Please enter a query to proceed.")
    else:
        st.warning("No text found in the PDF.")
