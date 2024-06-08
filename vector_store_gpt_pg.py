import openai
import os
from llama_index.readers.google import GoogleDriveReader
import psycopg2
from sqlalchemy import make_url
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
import textwrap

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"
openai.api_key = os.environ["OPENAI_API_KEY"]

connection_string = "postgresql://postgres:password@localhost:5432"
db_name = "vector_db"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

url = make_url(connection_string)

# folder_id="<folder_id>"
documents = GoogleDriveReader(file_ids=["<file_ids>"]).load_data()
for document in documents:
    document.text = document.text.replace("\x00", "\uFFFD")
    print(document.doc_id)
    print(document.text)

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="My Drive",
    embed_dim=1536,  # openai embedding dimension
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents,
                                        storage_context=storage_context,
                                        show_progress=True)
query_engine = index.as_query_engine()
response = query_engine.query(
    "Does this file contain data that may be sensitive? Classify what sensitive data might be in this file and give it a sensitivity score from 0-100."
)
print(textwrap.fill(str(response), 100))
