import logging
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.google import GoogleDriveReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import MetadataMode

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# folder_id="<folder_id>"
file_ids = ["1jy9ifuJbwGaE0PyZupA-a_NlqrF7yvi4XUbAZDMAwEw"]
documents = GoogleDriveReader().load_data(file_ids=file_ids)

# Documentation: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
ollama_embedding = OllamaEmbedding(
    model_name="llama3",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={
        "mirostat": 0,
        "num_ctx": 4096,
        "num_predict": -1,
        "seed": 42,
    },
)
ollama_llm = Ollama(model="llama3", request_timeout=120.0)
Settings.embed_model = ollama_embedding
Settings.llm = ollama_llm

for i in range(len(documents)):
    documents[i].text_template = """-----BEGIN METADATA-----
{metadata_str}
-----END METADATA-----
-----BEGIN CONTENT-----
{content}
-----END CONTENT-----"""

    print("The LLM sees this:\n" + documents[i].get_content(metadata_mode=MetadataMode.LLM))
    print("The Embedding model sees this:\n" + documents[i].get_content(metadata_mode=MetadataMode.EMBED))
    print(documents[i].__repr__)

# <{{ Potential PGVectorStore index
# connection_string = "postgresql://postgres:password@localhost:5432"
# db_name = "vector_db"
# conn = psycopg2.connect(connection_string)
# conn.autocommit = True

# with conn.cursor() as c:
#     c.execute(f"DROP DATABASE IF EXISTS {db_name}")
#     c.execute(f"CREATE DATABASE {db_name}")

# url = make_url(connection_string)

# vector_store = PGVectorStore.from_params(
#     database=db_name,
#     host=url.host,
#     password=url.password,
#     port=url.port,
#     user=url.username,
#     table_name="My Drive",
# )
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents,
#                                         storage_context=storage_context,
#                                         show_progress=True)
# }}>

# download and install dependencies
index = VectorStoreIndex.from_documents(
    documents=documents,
    show_progress=True,
)
query_engine = index.as_query_engine()

# set Logging to DEBUG for more detailed outputs
while True:
    query = input("Query: ")
    answer = query_engine.query(query)
    print(f"Answer: {answer}")
