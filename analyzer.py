from enum import Enum
from typing import List

import outlines
import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer


# Define enums for constrained fields
class DamageSeverity(str, Enum):
    NONE = "aucun"
    LIGHT = "léger"
    MODERATE = "modéré"
    SEVERE = "grave"


# Define the output structure using Pydantic
class Damage(BaseModel):
    type_dommage: str
    gravite: DamageSeverity
    piece: str


class DamageAnalysis(BaseModel):
    dommages: List[Damage]


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

    def prompt(self, description: str):
        messages = [
            {
                "role": "system",
                "content": "Tu es un expert en analyse de sinistres automobiles."
                "Ta tâche est d'extraire les informations clées des descriptions de dommages.",
            },
            {
                "role": "user",
                "content": f"""# Instructions: 
Analyse la description suivante et fournis une réponse structurée avec une liste de dommages. 
Pour chaque dommage, fournis le type de dommage, sa gravité (aucun/léger/modéré/grave) et la pièce du véhicule affectée.

Le format de la réponse doit être un JSON valide, avec le schéma suivant : 
 {{
    "dommages": [
        {{
            "type_dommage": <Type de dommage (par exemple : "impact avant gauche", "rayure", “aucun”, ...).>,
            "gravite": <Gravité du dommage (parmi : "léger", "modéré", "grave", “aucun”).>,
            "piece": <Estimation de la pièce du véhicule affectée (par exemple : "pare-chocs avant", "portière avant gauche", ...).>
        }}
    ]
}}

# Example: 
Exemple de description: 
Le pare-chocs avant est fissuré en plusieurs endroits avec un enfoncement sur le côté droit. La calandre est légèrement déformée mais reste fixée. Aucun dommage apparent sur les phares.

Exemple de sortie attendue:
{{
    "dommages": [
        {{
            "type_dommage": "Fissure, enfoncement",
            "gravite": "grave",
            "piece": "pare-chocs avant"
        }},
        {{
            "type_dommage": "aucun",
            "gravite": "aucun",
            "piece": "phare"
        }},
        {{
            "type_dommage": "déformation légère",
            "gravite": "leger",
            "piece": "calandre"
        }}
    ]
}}

# Input: 
Description du sinistre: {description}""",
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


if __name__ == "__main__":
    analyzer = DamageAnalyzer(device="cpu", seed=1024)
    print(
        analyzer.analyze(
            "Le pare-chocs arrière est totalement déformé suite à l'impact, mais en revoyant les dégâts, je remarque qu'il s'agit plutôt d'une fissure profonde au centre du pare-chocs.",
            "123",
        )
    )
