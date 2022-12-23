import os
import ask_embeddings
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")


query = "Tell me about 3P"
embeddings_file = "out/embeddings.pkl"

print(ask_embeddings.ask(query, embeddings_file))