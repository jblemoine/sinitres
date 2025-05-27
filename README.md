# Installation

## Python version

At least Python 3.11 is required.

## Install dependencies

Preferably install in a virtual environment, using `venv` or [uv](https://docs.astral.sh/uv/).


```bash
pip install .
```

# Hardware requirements

Prefer GPU, but CPU is also supported (slower). VRAM requirements are dependent on the model used.

# Usage 

```python
from analyzer import DamageAnalyzer

analyzer = DamageAnalyzer(model_name="google/gemma-3-1b-it") # or any other text generation model from https://huggingface.co/models
result = analyzer.analyze(description="The front bumper is cracked in several places with a dent on the right side. The grille is slightly deformed but remains attached. No apparent damage to the headlights.")

print(result)

# Output:
# DamageAnalysis(
#     damages=[
#         Damage(damage_type="crack, dent", severity="severe", part="front bumper"),
#         Damage(damage_type="light deformation", severity="light", part="grille"),
#     ]
# )
```

# Contributing

Install dependencies for dev

```bash
pip install -e .[dev]
```

###  Formatting and linting

Formatting and linting are done with [ruff](https://github.com/astral-sh/ruff).

```bash
ruff check . --fix
```

```bash
ruff format .
```

###  Type checking

Type checking is done with mypy.

```bash
mypy .
```

###  Testing

Test are run with pytest.

```bash
python -m pytest
```
###  Pre-commit

Pre-commit is used to run the linter and formatter before each commit.

```bash
# install pre-commit hooks
pre-commit install

# run the hooks
pre-commit run --all-files
```

## Context 

This project is a simple example of how to use a LLM to analyze a description of a car accident and return a structured output.
It was made as a proof of concept for an interview, and is not intended to be used in production.


## Presentation and answers to questions for the interview

• **Choice of model and solution architecture, training, inference and license.**

I chose the combination of a text generation model and a structured data generation library because it doesn't require pre-training and output format adherence is guaranteed.
It's compatible with any text generation model from HuggingFace, which allows to easily test different models.

During my testing I opted for gemma-3-1b-it because it's a very small model (1B parameters), with good performance, especially on multilingual tasks (in French in our case, the original usecase was in French). It's also memory efficient, thanks its K-V cache optimized architecture. (I initially tested with Phi-3.5-mini-instruct, but it required 20GB of VRAM, which is quite substantial for such a small model.)

I chose the outlines library because it enables structured JSON generation from LLMs with guaranteed schema compliance. Key advantages include:

1. **Zero overhead sampling**: Outlines pre-builds an index that maps valid tokens for each step of generation, allowing it to enforce structure without slowing down the model. This means generating structured JSON is almost just as fast as generating regular text.

2. **100% schema adherence**: Unlike libraries like instructor that use retry mechanisms, outlines guarantees valid output on the first attempt through logits masking during generation.

3. **Efficient token transitions**: The library uses "coalescence" - a technique that factorizes the generation process, allowing to skip decoding tokens that can be known in advance, e.g. keys in a JSON object, and then potentially speeding up the inference.

For the prompt, I opted for a classic prompt with a system message and a user message.
I didn't use very advanced techniques given the time constraints. Nevertheless, I made sure to be as precise as possible and to provide context and examples.

• **Proposal for technical and/or operational metrics for model evaluation**

Evaluation is challenging because in our case there is no exhaustive list of damage types or affected parts.

We could limit ourselves to a restricted number of damages and affected parts. Then establish a multi-label classification metric such as F1-score, Hamming Loss, etc.

We could also initially try to match predicted damages with actual damages, then calculate the percentage of correctly predicted damages using a similarity function.

• **Realistic improvement proposals if this project were to be developed in a business context:**

A larger dataset would be necessary to evaluate the model's generalization.
Other models, prompts, or potentially other solutions should be tested if the results are not satisfactory.

We could also fine-tune (lora) the model on a dataset. A few hundred examples might be sufficient to greatly improve performance.

• **What points of attention?**

I had some difficulties getting the model to work on GPU. However, I tested the model on CPU and it works correctly. I leave the choice to the user to select the device and LLM following the HuggingFace Transformers nomenclature.

• **How to deploy and what power is required?**

The model should be deployed on a optimized framework like VLLM, SGLang, or TensorRT, that can handle the KV cache efficiently across multiple requests.
Hopefully, outlines will be compatible with these frameworks.


References :
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- https://blog.dottxt.co/coalescence.html