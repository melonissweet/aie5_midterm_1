from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

env = load_dotenv()

qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")

path="data/"
loader = PyPDFDirectoryLoader(path, glob="*.pdf")
pdf_docs = loader.load()
print("Loaded PDFs")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

pdf_split_documents = text_splitter.split_documents(pdf_docs)
print(f"{len(pdf_split_documents)} split docs")

model_kwargs = {'trust_remote_code': True}
huggingface_embeddings = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l",
                                       model_kwargs=model_kwargs)
print("Loaded embeddings")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)
print("Initialize Qdrant client")

# qdrant_client.delete_collection("ontario_ltb_toronto_guidance")
# print("Deleted collection")

qdrant_client.create_collection(
    collection_name="ontario_ltb_toronto_guidance",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE), #size of embedding dimensions
)
print("Created collection")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="ontario_ltb_toronto_guidance",
    embedding=huggingface_embeddings,
)

vector_store.add_documents(documents=pdf_split_documents)

print("Documents added to vector store")