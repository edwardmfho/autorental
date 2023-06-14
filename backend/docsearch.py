import os

import pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


class EmbeddingPipeline:
  def __init__(self, 
               path_url = None, 
               online_pdf = True,
               chunk_size=2000, 
               chunk_overlap=0,
               is_update=False):

    # Containers
    self.pdfs = list()
    self.loaded_pdfs = list()
    self.splited_text = list()
    # Configurations
    self.online= online_pdf
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self.PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    self.PINECONE_ENVIRON = os.environ["PINECONE_ENVIRON"] 
    self.PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"] 

    self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    # Initialise instances
    pinecone.init(api_key=self.PINECONE_API_KEY,
                  environment=self.PINECONE_ENVIRON)    
    
    self.embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
    self.path_urls = path_url
    self.llm = OpenAI(temperature=0, openai_api_key=self.OPENAI_API_KEY)
    self.chain = load_qa_chain(self.llm, chain_type="stuff")
    self.index = pinecone.Index(self.PINECONE_INDEX_NAME)

    self.is_update = is_update
    self.docsearch = Pinecone.from_existing_index(self.PINECONE_INDEX_NAME, self.embeddings)
  # Functions

  # Utils
  def get_online_pdf(self, url: str) -> object:
    return OnlinePDFLoader(url)

  # Steps:
  # 01. Download and load PDF 
  # 02. Split text into chunk for each pdf 
  # 03. 
  # 
  def retreive_pdfs(self, online=True):
    for count, x in enumerate(self.path_urls):
      if online:
        print(f'{count} / Retrieving PDF from url: {x}')
        retrieved_pdf = self.get_online_pdf(x)
      self.pdfs.append(retrieved_pdf)
      print(f'Retrieved PDF {count}')

  def load_pdf(self, loader: object) -> object:
    data = loader.load()
    print (f'You have {len(data)} document(s) in your data')
    return data
  
  def load_multiple_pdf(self, pdfs=None):
    if pdfs is None:
      pdfs = self.pdfs
      self.loaded_pdfs = [self.load_pdf(pdf) for pdf in pdfs]

  def split_text(self, data: object) -> object:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                  chunk_overlap=self.chunk_overlap)
    texts = text_splitter.split_documents(data)
    return texts

  def insert_to_pinecone(self, texts):
    self.docsearch = Pinecone.from_texts([t.page_content for t in texts], self.embeddings, index_name=self.PINECONE_INDEX_NAME)

  def create_embeddings(self):
    if self.online:
      self.retreive_pdfs()
      self.load_multiple_pdf()

      for pdf_loader in self.loaded_pdfs:
        texts = self.split_text(pdf_loader)
        self.splited_text.append(texts)
      
      for text in self.splited_text:
        self.insert_to_pinecone(text)

  def query(self, query: str):
    if self.docsearch is not None:
      docs = self.docsearch.similarity_search(query)
      return self.chain.run(input_documents=docs, question=query)

