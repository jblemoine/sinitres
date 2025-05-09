# Installation

Preferably install in a virtual environment, using `venv` or [uv](https://docs.astral.sh/uv/).

```bash
pip install . --no-build-isolation # --no-build-isolation is necessary for flash-attn
```

# Hardware requirements

Prefer GPU, but CPU is also supported (slower). It will requires approximately 20GB of VRAM.

# Usage 

```python
from analyzer import DamageAnalyzer

analyzer = DamageAnalyzer(model_name="google/gemma-3-1b-it")
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

###  Testing

Test are run with pytest.

```bash
python -m pytest
```

###  Formatting and linting

Formatting and linting are done with [ruff](https://github.com/astral-sh/ruff).

```bash
ruff check . --fix
```

```bash
ruff format .
```

###  Pre-commit

Pre-commit is used to run the linter and formatter before each commit.

```bash
pre-commit run --all-files
```

## Context 

This project is a simple example of how to use a LLM to analyze a description of a car accident and return a structured output.
It was made as a proof of concept for an interview, and is not intended to be used in production.


## Presentation and answers to questions for the interview

• **Choice of model and solution architecture, training, inference and license.**

I chose to combine the Phi-3.5-mini-instruct LLM and the outlines library for structured data generation.

I chose this approach because it doesn't require pre-training and is very effective.
I opted for Phi-3.5-mini-instruct because it's a very small model (3B parameters), with good performance, especially on multilingual tasks (in French in our case, the original usecase was in French). According to benchmarks, its performance is very close to 7/8B-sized models.
The model's MIT license is very permissive and compatible with commercial use.

I chose the outlines library because it allows generating structured data using LLMs. Additionally, with outlines the inference time is almost not extended and adherence to the provided schema is 100% guaranteed (unlike other libraries like instructor, which under the hood uses a retry mechanism).

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

The model should be deployed on a server with GPU to enable inference with reasonable latency. It would also be interesting to test optimized versions of the model, particularly quantized versions (using the llamacpp library for example). The version presented here requires 20GB of VRAM, which is quite substantial for such a small model.
The model should be deployed using an API and a library capable of best managing batches like vllm, tgi, or TensorRT.

• **If you had domain experts available, how would you use them?**

I could ask them to qualitatively evaluate the model's results. I could also ask them to provide example data to evaluate the model's generalization.
I could also understand their operational mode and transcribe the information into the prompt.

• **How to monitor that the model doesn't drift over time?**

The model should be monitored using recent test data and by monitoring predictions, comparing results with predictions made on previous data.
For this, recurring tests can be performed, for example daily/weekly/monthly.


References :
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- https://blog.dottxt.co/coalescence.html