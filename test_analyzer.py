from analyzer import Damage, DamageAnalyzer


def test_damage_analyzer():
    description = "The rear bumper is completely deformed due to the impact."
    analyzer = DamageAnalyzer(model_name="google/gemma-3-1b-it")
    result = analyzer.analyze(description)
    assert result.damages == [
        Damage(damage_type="deformation", severity="severe", part="rear bumper")
    ]
