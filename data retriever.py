# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:10:09 2023

@author: dreji18
"""

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path=r'D:\Personal Projects\gpt4all\Marvel Datastore.csv', source_column="Sentences")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

question = "thor has any romantic relationship?"
docs = vectorstore.similarity_search(question, k=5)
len(docs)

docs[0].page_content


for i in docs:
    print(i)

#%%













