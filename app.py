import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import streamlit as st
import logging

from data.employees import generate_employee_data

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

if __name__ == "__main__":

    st.set_page_config(page_title="Company onboarding", page_icon="☂︎", layout="wide")

    logging.basicConfig(level=logging.INFO)

    @st.cache_data(ttl=3600, show_spinner="Loading employee data...")
    def get_user_data():
        return generate_employee_data(1)[0]
    
    @st.cache_resource(ttl=3600, show_spinner="Loading vector store...")
    def init_vector_store(pdf_path):
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1536,
                chunk_overlap = 200
            )

            chunks = text_splitter.split_documents(docs)

            embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            persistent_path = "./data/vectorestore"

            vectorstore = Chroma.from_documents(
                documents = chunks,
                embedding = embedding_model,
                persistent_path = persistent_path
            )

            return vectorstore

        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vectore store: {str(e)}")
            return None
    
    user_data = get_user_data()
    init_vector_store("data/umbrella_corp_policies.pdf")