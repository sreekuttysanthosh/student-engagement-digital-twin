
import pandas as pd
import pytest

from engagement_twin import generators
from engagement_twin.generators import TABLE, unwrap


def test_unwrap_dataframe_passthrough():
    df = pd.DataFrame({"a": [1]})
    assert unwrap(df) is df


def test_unwrap_dict_with_table_key():
    df = pd.DataFrame({"a": [1]})
    assert unwrap({TABLE: df}) is df


def test_unwrap_rejects_unexpected_table_key():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="Expected table"):
        unwrap({"other": df})


def test_unwrap_rejects_empty_dictionary():
    with pytest.raises(ValueError, match="Expected table"):
        unwrap({})


def test_unwrap_rejects_invalid_return_type():
    with pytest.raises(TypeError, match="DataFrame"):
        unwrap({"students": "invalid data"})





class MockSynthesizer:
    """Lightweight synthesizer for testing sampling functions."""

    def sample(self, num_rows):
        return pd.DataFrame({
            "nevents": range(num_rows),
            "Engagement_Level": [0] * num_rows,
        })

    def sample_from_conditions(self, conditions):
        condition = conditions[0]
        label = condition.column_values["Engagement_Level"]

        return pd.DataFrame({
            "nevents": range(condition.num_rows),
            "Engagement_Level": [label] * condition.num_rows,
        })


def test_load_generator_uses_saved_model(monkeypatch, tmp_path):
    model_path = tmp_path / "mock_model.pkl"
    model_path.touch()

    expected_model = object()

    def fake_load(path):
        assert path == str(model_path)
        return expected_model

    monkeypatch.setattr(generators, "load_synthesizer", fake_load)

    result = generators.load_generator(model_path)

    assert result is expected_model


def test_load_generator_rejects_missing_file(tmp_path):
    model_path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError):
        generators.load_generator(model_path)


def test_sample_unconditional_returns_requested_rows():
    synth = MockSynthesizer()

    result = generators.sample_unconditional(
        synth,
        n_rows=100,
        seed=42,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 100


def test_sample_unconditional_rejects_zero_rows():
    synth = MockSynthesizer()

    with pytest.raises(ValueError, match="greater than zero"):
        generators.sample_unconditional(
            synth,
            n_rows=0,
            seed=42,
        )


def test_sample_balanced_returns_equal_classes():
    synth = MockSynthesizer()

    result = generators.sample_balanced(
        synth,
        target="Engagement_Level",
        n_per_class=50,
        seed=42,
    )

    counts = result["Engagement_Level"].value_counts().to_dict()

    assert len(result) == 100
    assert counts == {0: 50, 1: 50}


def test_sample_balanced_rejects_zero_rows():
    synth = MockSynthesizer()

    with pytest.raises(ValueError, match="greater than zero"):
        generators.sample_balanced(
            synth,
            target="Engagement_Level",
            n_per_class=0,
            seed=42,
        )
