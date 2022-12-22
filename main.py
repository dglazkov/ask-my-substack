import os
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import numpy as np
from transformers import GPT2TokenizerFast


MODEL_NAME = "text-embedding-ada-002"
SEPARATOR = "\n"
MAX_PROMPT_LEN = 2048

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")


def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))


def get_embedding(text: str) -> list[float]:
    result = openai.Embedding.create(
        model=MODEL_NAME,
        input=text
    )
    return result["data"][0]["embedding"]


def strip_emoji(text):
    import re
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


with open("in/posts/36704568.2021-05-17.html", "r") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
embedings = []
for sibling in soup.children:
    text = strip_emoji(sibling.get_text(" ", strip=True))
    embedding = get_embedding(text)
    embedings.append((text, embedding))

query = "why do you call them crappy?"
query_embedding = get_embedding(query)

similiarities = sorted([
    (vector_similarity(query_embedding, embedding), text) for text, embedding in embedings], reverse=True)

# for similarity, text in similiarities:
#     print(similarity, text)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

context = []
context_len = 0

for similarity, text in similiarities:
    text_token_len = len(tokenizer.tokenize(text))
    context_len += text_token_len + separator_len
    if context_len > MAX_PROMPT_LEN:
        if len(context) == 0:
            context.append(text[:(MAX_PROMPT_LEN - separator_len)])
        break
    context.append(text)

# print(context)

prompt = f"Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say \"I don't know.\"\n\nContext:\n{context} \n\nQuestion:\n{query}\n\nAnswer:"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(response.choices[0].text.strip())
