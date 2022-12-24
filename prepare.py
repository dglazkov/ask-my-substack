import glob
import os
import pickle
import re

import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast

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
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def get_issue_slug(file_name):
    match = re.search(r"(?<=\.)[^.]*(?=\.)", file_name)
    if match:
        return match.group()
    return None


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
embeddings = []
issue_info = {}

html_files = glob.glob("in/posts/*.html")

for index, html_file in enumerate(html_files):
    print(f"Processing {html_file}...")
    issue_slug = get_issue_slug(html_file)
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, "html.parser")
        img = soup.find("img", recursive=True)
        if img is not None:
            img = img["src"]
        issue_info[index] = (issue_slug, img)
        for sibling in soup.children:
            text = strip_emoji(sibling.get_text(" ", strip=True))
            if len(text) < USELESS_TEXT_THRESHOLD:
                continue
            embedding = get_embedding(text)
            embeddings.append((text, embedding, len(tokenizer.tokenize(text)), index))

with open('out/embeddings.pkl', 'wb') as f:
    pickle.dump({
        "embeddings": embeddings,
        "issue_info": issue_info
    }, f)

print("Done!")
