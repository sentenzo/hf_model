# https://huggingface.co/TheBloke/rocket-3B-GGUF

import os

import outlines
from dotenv import load_dotenv

load_dotenv()

model = outlines.models.llamacpp(os.getenv("PATH_TO_MODEL", ""))
prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
print(answer)
