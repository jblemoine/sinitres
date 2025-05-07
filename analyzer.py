from enum import Enum

import outlines
import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer


class DamageSeverity(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"


class Damage(BaseModel):
    type_dommage: str
    gravite: DamageSeverity
    piece: str


class DamageAnalysis(BaseModel):
    dommages: list[Damage]


class DamageAnalyzer:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int | None = None,
    ):
        self.model = outlines.models.transformers(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = outlines.generate.json(self.model, DamageAnalysis)
        self.seed = seed

    def prompt(self, description: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an expert in analyzing car accident descriptions."
                "Your task is to extract the key information from the damage descriptions.",
            },
            {
                "role": "user",
                "content": f"""# Instructions: 
Analyze the following description and provide a structured response with a list of damages. 
For each damage, provide the type of damage, its severity (none/light/moderate/severe) and the affected vehicle part.

The format of the response must be a valid JSON, with the following schema: 
 {{
    "dommages": [
        {{
            "type_dommage": <Type of damage (for example: "left front impact", "scratch", "none", ...).>,
            "gravite": <Severity of the damage (either: "light", "moderate", "severe", "none").>,
            "piece": <Estimation of the affected vehicle part (for example: "front bumper", "left door", ...).>
        }}
    ]
}}

Input: The front bumper is cracked in several places with a dent on the right side. The grille is slightly deformed but remains attached. No apparent damage to the headlights.
Output:
{{
    "dommages": [
        {{
            "type_dommage": "Crack, dent",
            "gravite": "severe",
            "piece": "front bumper"
        }},
        {{
            "type_dommage": "none",
            "gravite": "none",
            "piece": "headlight"
        }},
        {{
            "type_dommage": "light deformation",
            "gravite": "light",
            "piece": "grille"
        }}
    ]
}}
Input: {description}
Output:
""",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_bos=True, add_generation_prompt=True
        )

        return prompt

    def analyze(self, description: str, sinistre_id: str) -> pd.DataFrame:
        """
        Analyze the damage description and return a list of damages as a pandas DataFrame.
        """
        damages = self.generator(self.prompt(description), seed=self.seed).dommages
        df = pd.DataFrame([damage.model_dump() for damage in damages])
        df["sinistre"] = sinistre_id
        return df
