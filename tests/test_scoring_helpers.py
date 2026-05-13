from pathlib import Path

import pandas as pd

from promptstability.core import PromptStabilityAnalysis


def dummy_annotation_function(text, prompt):
    return 0


def make_psa():
    return PromptStabilityAnalysis(
        annotation_function=dummy_annotation_function,
        data=[],
        load_generation_models=False,
    )


def test_score_intra_annotations_supports_cumulative_and_adjacent():
    psa = make_psa()
    fixture_path = Path("tests/annotations/test_intra_annotations.csv")
    annotated_df = pd.read_csv(fixture_path)

    score_map, rescored = psa.score_intra_annotations(
        annotated_df,
        bootstrap_samples=20,
        analysis_modes=["cumulative_alpha", "adjacent_alpha"],
    )

    assert set(score_map.keys()) == {"cumulative_alpha", "adjacent_alpha"}
    assert "cumulative_ka_mean" in rescored.columns
    assert "adjacent_ka_mean" in rescored.columns
    assert "ka_mean" in rescored.columns
    assert len(score_map["cumulative_alpha"]) == annotated_df["iteration"].nunique() - 1
    assert len(score_map["adjacent_alpha"]) == annotated_df["iteration"].nunique() - 1


def test_summarize_intra_scores_returns_both_summary_blocks():
    psa = make_psa()
    fixture_path = Path("tests/annotations/test_intra_annotations.csv")
    annotated_df = pd.read_csv(fixture_path)

    score_map, _ = psa.score_intra_annotations(
        annotated_df,
        bootstrap_samples=20,
        analysis_modes=["cumulative_alpha", "adjacent_alpha"],
    )
    summaries = psa.summarize_intra_scores(score_map)

    assert "cumulative_alpha" in summaries
    assert "adjacent_alpha" in summaries
    assert "final_alpha" in summaries["cumulative_alpha"]
    assert "run_count_to_estimate_stability" in summaries["cumulative_alpha"]
    assert "mean_alpha" in summaries["adjacent_alpha"]
    assert "share_below_threshold" in summaries["adjacent_alpha"]


def test_extract_and_summarize_inter_scores_from_saved_annotations():
    psa = make_psa()
    fixture_path = Path("tests/annotations/test_inter_annotations.csv")
    annotated_df = pd.read_csv(fixture_path)

    # Mirror the package's inter-PSS output columns from the saved fixture.
    score_map = {}
    for temperature in sorted(annotated_df["temperature"].unique()):
        subset = annotated_df[annotated_df["temperature"] == temperature]
        mean_alpha, (ci_lower, ci_upper) = psa.bootstrap_krippendorff(
            subset,
            annotator_col="prompt_id",
            bootstrap_samples=20,
        )
        score_map[temperature] = {
            "Average Alpha": mean_alpha,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
        }
        annotated_df.loc[annotated_df["temperature"] == temperature, "ka_mean"] = mean_alpha
        annotated_df.loc[annotated_df["temperature"] == temperature, "ka_lower"] = ci_lower
        annotated_df.loc[annotated_df["temperature"] == temperature, "ka_upper"] = ci_upper

    extracted = psa.extract_inter_score_map(annotated_df)
    summaries = psa.summarize_inter_scores(extracted)

    assert set(extracted.keys()) == set(score_map.keys())
    assert "mean_alpha" in summaries
    assert "temperature_at_min_alpha" in summaries
    assert "share_temperatures_below_threshold" in summaries
