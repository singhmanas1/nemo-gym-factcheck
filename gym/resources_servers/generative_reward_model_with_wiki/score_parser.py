import re
from typing import List, Tuple


def _parse_string_value(raw: str) -> Tuple[List[float], bool]:
    """Parse a string like '2, 4' or '6' into list of floats. Returns (scores, all_valid). Invalid tokens become 0.0."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    scores: List[float] = []
    for p in parts:
        try:
            scores.append(float(p))
        except (ValueError, TypeError):
            all_valid = False
            return [0.0, 0.0], False
    return scores, True


def extract_scores(text: str, num_responses: int) -> Tuple[List[float], float, bool]:
    """
    Extract individual scores and ranking score from text.

    num_responses: 1 or 2.
    - If 1: expect [The Begin of Individual Scores] ... \\boxed{<score_1>} ... only; no ranking.
    - If 2: expect \\boxed{<score_1>, <score_2>} and [The Begin of Ranking Score] ... \\boxed{<ranking>} ...

    Returns:
        (individual_scores, ranking_score, format_correct)
        - individual_scores: always length 2, e.g. [2.0, 4.0] or [3.0, 0.0] for single response; [0.0, 0.0] if invalid
        - ranking_score: float; 0.0 if invalid or num_responses==1
        - format_correct: True if format matches expected for num_responses
    """
    if num_responses not in (1, 2):
        raise ValueError(f"num_responses must be 1 or 2, got {num_responses}")

    boxed = re.compile(r"\\boxed\{([^}]+)\}")

    individual_section = re.search(
        r"\[The Begin of Individual Scores\].*?\[The End of Individual Scores\]",
        text,
        re.DOTALL,
    )
    ranking_section = re.search(
        r"\[The Begin of Ranking Score\].*?\[The End of Ranking Score\]",
        text,
        re.DOTALL,
    )

    individual_scores: List[float] = []
    ranking_score: float = 0.0
    format_correct = False

    ind_match = boxed.search(individual_section.group(0)) if individual_section else None
    ind_valid = True
    if ind_match:
        individual_scores, ind_valid = _parse_string_value(ind_match.group(1))

    expected_count = num_responses
    if len(individual_scores) != expected_count:
        individual_scores = [0.0, 0.0]
        format_correct = False
    else:
        format_correct = ind_valid
        # Pad to length 2 so app can always use [0] and [1]
        if num_responses == 1:
            individual_scores = [individual_scores[0], 0.0]

    if num_responses == 1:
        return individual_scores, 0.0, format_correct

    rank_match = boxed.search(ranking_section.group(0)) if ranking_section else None
    if rank_match:
        values, rank_valid = _parse_string_value(rank_match.group(1))
        if len(values) == 1 and rank_valid:
            ranking_score = values[0]
        else:
            ranking_score = 0.0
            format_correct = False
    else:
        format_correct = False

    return individual_scores, ranking_score, format_correct


if __name__ == "__main__":
    sample_two = """
[The Begin of Analysis on Response 1]
Response 1 claims "1+2=4"...
[The End of Analysis on Response 1]

[The Begin of Individual Scores]
\\boxed{2.5, 4}
[The End of Individual Scores]

[The Begin of Ranking Score]
\\boxed{6}
[The End of Ranking Score]
"""
    individual, ranking, format_ok = extract_scores(sample_two, num_responses=2)
    print("Two responses:", individual, ranking, format_ok)  # [2.0, 4.0], 6.0, True

    sample_one = """
[The Begin of Analysis on Response 1]
Analysis of the single response.
[The End of Analysis on Response 1]

[The Begin of Individual Scores]
\\boxed{3}
[The End of Individual Scores]
"""
    individual, ranking, format_ok = extract_scores(sample_one, num_responses=1)
    print("One response:", individual, ranking, format_ok)  # [3.0, 0.0], 0.0, True