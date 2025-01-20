# Installation

Preferably install in a virtual environment, using `venv` or [uv](https://docs.astral.sh/uv/).

```bash
pip install . 
```

# Hardware requirements

Prefer GPU, but CPU is also supported (slower). It will requires approximately 20GB of VRAM.

# Usage 

```python
from analyzer import DamageAnalyzer

analyzer = DamageAnalyzer(device="cpu")
result = analyzer.analyze("Le pare-chocs avant est fissuré en plusieurs endroits avec un enfoncement sur le côté droit. La calandre est légèrement déformée mais reste fixée. Aucun dommage apparent sur les phares.", "123")

print(result)
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


## Présentation et réponse aux questions

• **Choix du modèle et de l’architecture de la solution, entrainement, inférence et licence.**

J'ai choisi de combiner le LLM Phi-3.5-mini-instruct et la librairie outlines pour la génération de données structurées.

J'ai choisi cette approche car elle ne demande pas de faire de pré-entrainement et est très performante.
J'ai opté pour Phi-3.5-mini-instruct car c'est un très petit modèle (3B de paramètres), avec de bonnes performances, notamment sur les tâches multilangues (en français dans notre cas). D'après les benchmarks, il est très proche des modèles de taille 7/8B.
La licence MIT du modèle est très permissive et est compatible avec une utilisation commerciale.

J'ai opté pour la librairie outlines car elle permet de générer des données structurées en utilisant des LLM. En outre avec outlines le temps d'inférence n'est pas ralongé et l'adéquation au schéma fourni est assurée à 100% (contrairement à d'autres librairies comme instructor).

Côté prompt, j'ai opté pour un prompt classique avec un système message et un user message.
Je n'ai pas utilisé de technique très avancées compte tenu du temps imparti. J'ai tout de même veillé à être aussi précis possible et à lui fournir du contexte et des exemples.

• **Proposition de métriques techniques et/ou opérationnelles d’évaluation du modèle**
L'évaluation est délicate car il n'existe pas notre cas une liste exhaustive de type de dommages ou de pièces affectées.

On pourrait se limiter à un nombre restreint de dommages et de pièces affectées. Puis établir une métrique de classification multi-label comme le F1-score, la Hamming Loss, etc.


On pourrait également essayer dans un premier de faire coincider les dommages prédits avec les dommages réels, puis de calculer le pourcentage de dommages prédits correctement à l'aide d'une fonction de similarité.

• **Propositions d’amélioration réalistes si ce projet devait se développer en d’entreprise :**

Un jeu de données plus large serait nécessaire pour évaluer la généralisation du modèle.
Il faudrait tester d'autres modèles, d'autres prompt, ou éventuellement d'autres solutions si les résultats ne sont pas satisfaisants. 

**◦ Quels points d’attention ?**
J'ai eu quelques difficultés pour faire fonctionner le modèle sur GPU. Cependant j'ai testé le modèle sur CPU et il fonctionne correctement. Je laisse le choix à l'utilisateur de choisir le device et le LLM en suivant la 

**◦ Comment déployer et quelle puissance requise ?**
Il faudrait déployer le modèle sur un serveur avec GPU, afin de pouvoir faire des inférences avec un temps de latence raisonnable. Il serait également intéressant de tester des versions optimisées du modèle, notamment des versions quantisées (à l'aide de la librairie llamacpp par exemple). La version présentée ici nécessite 20GB de VRAM, ce qui est tout de même assez conséquent pour un modèle de taille si petite.
Il faudrait déployer le modèle à l'aide d'une API et d'une librairie capable de gérer au mieux les batch comme vllm, tgi ou triton.

**◦ Si vous aviez des experts métier à disposition comment les utiliseriez-vous ?**
Je pourrais leur demander d'évaluer qualitativement les résultats du modèle. Je pourrais également leur demander de fournir des exemples de données pour évaluer la généralisation du modèle.
Je pourrais égelement comprendre leur mode de fonctionnement et retranscrire les informations dans le prompt.

**◦ Comment surveiller que le modèle ne dévie pas dans le temps ?**
Il faudrait surveiller le modèle en utilisant des données de test récentes et en surveillant les prédictions, et en comparant les résultats avec des prédictions effectuées sur des données antérieures.
Pour cela on peut effectuer des tests récurrents, par exemple chaque jour / semaine / mois.



Références :
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- https://blog.dottxt.co/coalescence.html