"""Backward-compatible exports for the finalized paraphrase sets."""

from Prompts.prompt_sets import paraphrase_a_prompts, paraphrase_b_prompts


shadow_a_prompts = paraphrase_a_prompts
shadow_b_prompts = paraphrase_b_prompts

# Preserve the original module's individual public names. They now point to
# the finalized first paraphrase set instead of maintaining a third copy.
(
    shadow_prompt_1,
    shadow_prompt_2,
    shadow_prompt_3,
    shadow_prompt_4,
    shadow_prompt_5,
) = shadow_a_prompts
