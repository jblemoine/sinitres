from analyzer import DamageAnalyzer


def test_damage_analyzer():
    description = "The rear bumper is completely deformed due to the impact, but upon closer inspection, it appears to be a deep crack in the center of the bumper."
    analyzer = DamageAnalyzer(
        device="cpu", seed=100, model_name="microsoft/Phi-3.5-mini-instruct"
    )
    result = analyzer.analyze(description, sinistre_id="123")
    assert result.shape[0] > 0

    for row in result.itertuples():
        assert row.type_dommage
        assert row.gravite
        assert row.piece
