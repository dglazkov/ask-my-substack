import os
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
import pickle
import glob

MODEL_NAME = "text-embedding-ada-002"
USELESS_TEXT_THRESHOLD = 100

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")

def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
        model=MODEL_NAME,
        input=text
    )
    return result["data"][0]["embedding"]


def strip_emoji(text: str):
    import re
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
embeddings = []

html_files = glob.glob("in/posts/*.html")

for html_file in html_files:
  print(f"Processing {html_file}...")
  with open(html_file, 'r') as file:
    soup = BeautifulSoup(file, "html.parser")
    for sibling in soup.children:
        text = strip_emoji(sibling.get_text(" ", strip=True))
        if len(text) < USELESS_TEXT_THRESHOLD:
          continue
        embedding = get_embedding(text)
        embeddings.append((text, embedding, len(tokenizer.tokenize(text))))

with open('out/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Done!")