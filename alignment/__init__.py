from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import (
    answer_tag_reward_fn,
    extract_answer_from_tags,
    majority_vote_tagged_answers,
    r1_zero_reward_fn,
)

__all__ = [
    "COT_PROMPT_TEMPLATE",
    "DIRECT_PROMPT_TEMPLATE",
    "answer_tag_reward_fn",
    "extract_answer_from_tags",
    "majority_vote_tagged_answers",
    "r1_zero_reward_fn",
]
