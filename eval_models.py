import asyncio
import os
import openai
import weave
import io
import base64
from typing import Union
from PIL import Image
import instructor
from pydantic import BaseModel, Field
from datasets import load_from_disk
import simple_parsing
from dataclasses import dataclass

@dataclass
class Args:
    n: int = 100

args = simple_parsing.parse(Args)

HF_DATASET = "qa_german_invoices_formatted"

qa_ds = load_from_disk(HF_DATASET)

# Slice the first n elements of the dataset
qa_ds = qa_ds.select(range(min(args.n, len(qa_ds))))

print(f"Evaluating {len(qa_ds)} samples")

weave.init("german_invoices_eval")

# our own Llama 3.2-90B-Vision-Instruct instance
llama_client = openai.AsyncOpenAI(
    base_url="http://195.242.25.198:8032/v1",
    api_key=os.environ.get("WANDB_API_KEY"),
)


def image_to_base64(image_path: Union[str, Image]) -> str:
    image = Image.open(image_path) if isinstance(image_path, str) else image_path
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    return str(encoded_string)


questions_template = """
You need to extract the answers to the questions from the document.
Reply in the following format:
1. question: answer
2. question: answer
etc...
Here you have the questions:
{questions}
"""


class Model(weave.Model):
    client: openai.AsyncOpenAI
    model: str = "Llama-3.2-90B-Vision-Instruct"

    @weave.op
    async def predict(self, image: Image.Image, questions: list[str]) -> str:
        image_base64 = image_to_base64(image)
        questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        base64_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": questions_template.format(questions=questions),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=base64_messages,
        )
        return response.choices[0].message.content


llama = Model(client=llama_client, model="Llama-3.2-90B-Vision-Instruct")

print("Defining Judge")
openai_client = instructor.from_openai(openai.AsyncOpenAI())


class Judge(BaseModel):
    matches: bool = Field(description="Whether the model's answer matches the real answer")
    extracted_answer: str = Field(description="The extracted answer from the model")
    explanation: str = Field(
        description="The explanation for the correctness or incorrectness of the answer"
    )


system_prompt = """Your tasks is to determine if the extracted answer matches the real answer. \
Be tolerant to spelling mistakes that could be related to the OCR extraction of the documents and languages. \
We only care about the answers, not the questions itself.
Be tolerant about about the format of the answers, for example:
Mr. Dave is a valid answer for "Dave"
or "the 02/2024 invoice" is a valid answer for "Invoice from 2024-02"
We want to check if the model is retrieving the information to answer the questions correctly.
"""

prompt_template = """
## Model Extracted Answers
{model_output}
## Real Answers
{answers}

Reply in JSON format",
"""


@weave.op
async def extract(model_output: str, answers: str) -> dict:
    res = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_template.format(
                    model_output=model_output, answers=answers
                ),
            },
        ],
        response_model=list[Judge],
    )
    return res

@weave.op
async def judge_answer(model_output: str, answers: str) -> dict:
    res = await extract(model_output, answers)
    return {
        "correct": sum([r.matches for r in res]),
        "explanation": "\n".join([r.explanation for r in res]),
    }


# let's create a flat version of the dataset, with the questions stacked and answers as a list of strings
flat_ds = [
    {
        "image": sample["image"],
        "questions": [q["question"] for q in sample["qa_pairs"]],
        "answers": [q["answer"] for q in sample["qa_pairs"]],
    }
    for sample in qa_ds
]

evaluation = weave.Evaluation(dataset=flat_ds, scorers=[judge_answer])

print("=" * 100)
print("Evaluating Llama")
print("=" * 100)

asyncio.run(evaluation.evaluate(llama))

mistral_client = openai.AsyncOpenAI(
    base_url="https://api.mistral.ai/v1/",
    api_key=os.environ.get("MISTRAL_API_KEY"),
)

mistral = Model(client=mistral_client, model="pixtral-12b-2409")
print("=" * 100)
print("Evaluating Mistral")
print("=" * 100)

asyncio.run(evaluation.evaluate(mistral))

