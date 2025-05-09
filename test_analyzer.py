from analyzer import DamageAnalyzer


def test_damage_analyzer():
    description = "The rear bumper is severely deformed."
    analyzer = DamageAnalyzer(model_name="google/gemma-3-1b-it")
    result = analyzer.analyze(description)
    assert len(result.damages) == 1
