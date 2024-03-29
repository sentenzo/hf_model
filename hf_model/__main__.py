# https://huggingface.co/TheBloke/rocket-3B-GGUF

import os

from dotenv import load_dotenv
from outlines.generate import choice
from outlines.models import llamacpp

load_dotenv()

model_path = os.getenv("PATH_TO_MODEL", "")
model = llamacpp(
    model_path,
    model_kwargs={
        "n_gpu_layers": -1,
        "n_batch": 512,
        "n_ctx": 1000,
        "temp": 0.8,
    },
)
prompt_stub = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: """

inputs = [
    # Positive
    "This restaurant is just awesome!",
    "Great service and a wonderful dining experience!",
    "Highly recommend this place for its fantastic atmosphere!",
    "The staff was friendly and attentive, making our visit enjoyable.",
    "Amazing flavors and exceptional presentation, a top-notch restaurant!",
    # Negative
    "Terrible food quality and disappointing service.",
    "The restaurant was unclean and poorly maintained.",
    "Overpriced dishes that lacked flavor and creativity.",
    "Rude staff members ruined the dining experience.",
    "Long wait times and chaotic atmosphere, not worth a visit.",
]


generator = choice(model, ["Positive", "Negative"])
answers = []
for message in inputs:
    answer = generator(prompt_stub + message)
    answers.append(answer)
for message, answer in zip(inputs, answers):
    print(answer, "-", message)
