"""The LLM hypothesis-belief side job: elicit household descriptions
from an LLM, convert them (:mod:`baselines.beliefs.hypothesis_program`),
weigh them on sightings (:mod:`baselines.beliefs.llm_hypothesis_mixture`),
and evaluate on the belief benchmark. Belief-only; nothing here touches
decision policies."""
