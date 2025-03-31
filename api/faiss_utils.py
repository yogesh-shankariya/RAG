from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from db_utils import get_all_documents
import shutil
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_index_path = "./faiss_index"


# Load existing FAISS index or create a new one
if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, embedding_function)
else:
    vectorstore = FAISS.from_texts(["Initialize"], embedding_function)
    vectorstore.save_local(faiss_index_path)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata['file_id'] = file_id

        vectorstore.add_documents(splits)
        vectorstore.save_local(faiss_index_path)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def rebuild_faiss_index_excluding(file_id_to_delete: int):
    try:
        all_documents = get_all_documents()
        remaining_docs = [doc for doc in all_documents if doc['id'] != file_id_to_delete]

        # Delete existing index directory
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)

        # Reinitialize the vectorstore
        global vectorstore
        vectorstore = FAISS.from_texts(["Initialize"], embedding_function)

        for doc in remaining_docs:
            splits = load_and_split_document(doc['filename'])
            for split in splits:
                split.metadata['file_id'] = doc['id']
            vectorstore.add_documents(splits)

        vectorstore.save_local(faiss_index_path)
        print(f"FAISS index rebuilt successfully without file_id {file_id_to_delete}.")
        return True
    except Exception as e:
        print(f"Error rebuilding FAISS index: {e}")
        return False

