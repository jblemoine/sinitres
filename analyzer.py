from typing import Literal

from pydantic import BaseModel, Field


class Damage(BaseModel):
    damage_type: str = Field(description="The type of damage.")
    severity: Literal["light", "moderate", "severe"] | None = Field(
        description="The severity of the damage."
    )
    part: str = Field(description="The part of the vehicle that is damaged.")


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
        import torch
        from outlines.models import TransformerTokenizer
        from outlines.processors.structured import JSONLogitsProcessor
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LogitsProcessorList,
            pipeline,
        )

        quantization_config = (
            BitsAndBytesConfig(load_in_8bit=True) if torch.cuda.is_available() else None
        )
        # init model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
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
For each damage, provide the type of damage, its severity (light/moderate/severe/none) and the affected vehicle part.

The format of the response must be a valid JSON, with the following schema: 
{DamageAnalysis.model_json_schema()}

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
