"""Patrol-stream evaluation: a fixed robot patrol, one free look per
question, an ABSTAIN answer, and a hint channel for LLM agents.

Modules: ``bank`` builds a bank from a situation_sim run; ``run`` replays
it against classical agents; ``llm`` runs the LLM agents; ``summary``
pools run logs into the report tables; ``leak_check`` fails a run whose
prompts contain hidden state.
"""
