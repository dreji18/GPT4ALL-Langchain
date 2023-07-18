# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:06:50 2023

@author: dreji18
"""

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
import streamlit as st

from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

loader = CSVLoader(file_path=r'D:\Personal Projects\gpt4all\marvel app\Marvel Datastore.csv', source_column="Sentences")


st.title("Marvel bot")


data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create the vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# User input and document similarity
question = st.text_input("Ask a question about Marvel:")
if question:
    docs = vectorstore.similarity_search(question, k=1)

    if docs:
        st.markdown(docs[0].page_content.split("Sentences:")[-1])
