import json
from enum import Enum

from pydantic import BaseModel


class DamageSeverity(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"


class Damage(BaseModel):
    damage_type: str
    severity: DamageSeverity
    part: str


class DamageAnalysis(BaseModel):
    damages: list[Damage]


class DamageAnalyzer:
    """
    Analyze the damage description and return a list of damages.
    Args:
        model_name: The repo_id of the model to use, download from https://huggingface.co/models
    """

    def __init__(
        self,
        model_name: str,
    ):
        from outlines.models import TransformerTokenizer
        from outlines.processors.structured import JSONLogitsProcessor
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LogitsProcessorList,
            pipeline,
        )

        # init model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        # init tokenizer and logits processor
        self.outlines_tokenizer = TransformerTokenizer(self.tokenizer)
        self.outlines_logits_processor = JSONLogitsProcessor(
            schema=DamageAnalysis, tokenizer=self.outlines_tokenizer
        )
        self.logits_processor_list = LogitsProcessorList(
            [self.outlines_logits_processor]
        )

    def prompt(self, description: str) -> list[dict]:
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
    "damages": [
        {{
            "damage_type": <Type of damage (for example: "left front impact", "scratch", "none", ...).>,
            "severity": <Severity of the damage (either: "light", "moderate", "severe", "none").>,
            "part": <Estimation of the affected vehicle part (for example: "front bumper", "left door", ...).>
        }}
    ]
}}

# Example:
Description: The front bumper is cracked in several places with a dent on the right side. The grille is slightly deformed but remains attached.
Output:
{
                    json.dumps(
                        DamageAnalysis(
                            damages=[
                                Damage(
                                    damage_type="crack, dent",
                                    severity="severe",
                                    part="front bumper",
                                ),
                                Damage(
                                    damage_type="light deformation",
                                    severity="light",
                                    part="grille",
                                ),
                            ]
                        ).model_dump(mode="json")
                    )
                }

# Input:
Description: {description}
Output:
""",
            },
        ]

        return messages

    def analyze(self, description: str) -> DamageAnalysis:
        """
        Analyze the damage description and return a list of damages as a pandas DataFrame.
        """
        messages = self.prompt(description)
        completion = self.pipeline(
            messages,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            return_full_text=False,
            logits_processor=self.logits_processor_list,
        )
        analysis = DamageAnalysis.model_validate_json(completion[0]["generated_text"])
        return analysis
