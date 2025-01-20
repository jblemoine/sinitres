from analyzer import DamageAnalyzer


def test_damage_analyzer():
    description = "Le pare-chocs arrière est totalement déformé suite à l'impact, mais en revoyant les dégâts, je remarque qu'il s'agit plutôt d'une fissure profonde au centre du pare-chocs."
    analyzer = DamageAnalyzer(
        device="cpu", seed=100, model_name="microsoft/Phi-3.5-mini-instruct"
    )
    result = analyzer.analyze(description, sinistre_id="123")
    assert result.shape[0] > 0

    for row in result.itertuples():
        assert row.type_dommage
        assert row.gravite
        assert row.piece
