import asyncio
import os
import json
import openai
import weave
import io
import base64
import random
from typing import Union
from pathlib import Path
from PIL import Image
import simple_parsing
from dataclasses import dataclass

@dataclass
class Args:
    n: int = 100
    base_url: str = "http://195.242.25.198:8032/v1"
    api_key: str = os.environ.get("WANDB_API_KEY")
    seed: int = 42
    model: str = "Llama-3.2-90B-Vision-Instruct"
    weave: bool = True
    temperature: float = 0.5
    max_tokens: int = 100


args = simple_parsing.parse(Args)
if args.weave:
    weave.init("OCR_bench")


random.seed(args.seed)

data_path = Path("OCRBench")

ds_file = data_path / "OCRBench.json"

ds = json.load(ds_file.open())

print(f"Total dataset size: {len(ds)}")
# ds = random.sample(ds, min(args.n, len(ds)))
print(f"Evaluating {len(ds)} samples")
print("="*100)
print("Dataset Sample:")
print(ds[0])
print("="*100)

def image_to_base64(image_path: Union[str, Image, Path]) -> str:
    image = Image.open(image_path) if isinstance(image_path, str | Path) else image_path
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    return str(encoded_string)


# our own Llama 3.2-90B-Vision-Instruct instance
llama_client = openai.AsyncOpenAI(
    base_url=args.base_url,
    api_key=args.api_key,
)

questions_template = """
reply to the question based on the document. Reply with the answers only, don't add a period at the end of the answer.
question: {question}
answer:
"""


class Model(weave.Model):
    client: openai.AsyncOpenAI
    model: str = "Llama-3.2-90B-Vision-Instruct"
    data_path: Path = data_path
    temperature: float = args.temperature
    max_tokens: int = args.max_tokens

    @weave.op
    async def predict(self, image_path: str, question: list[str]) -> str:
        image_base64 = image_to_base64(self.data_path / image_path)
        base64_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": questions_template.format(question=question),
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
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {"extracted_answer": response.choices[0].message.content}

print(f"Testing model: {args.model}")
print("="*100)    
model = Model(client=llama_client, data_path=data_path, model=args.model)

def match_answer(model_output: str, answers: str) -> dict:
    try:
        return {
            "match": model_output["extracted_answer"].lower() in str(answers).lower(),
        }
    except Exception as e:
        print(f"Error matching answer: {e}")
        return {
            "match": False,
            "error": str(e),
        }

evaluation = weave.Evaluation(dataset=ds, scorers=[match_answer])
asyncio.run(evaluation.evaluate(model))
