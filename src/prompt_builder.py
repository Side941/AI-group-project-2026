"""
prompt_builder.py
=================
Builds prompts for suicide classification.
"""

from __future__ import annotations

from typing import List, Tuple
from src.config import (
    ZERO_SHOT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    EXAMPLE_TEMPLATE,
    get_labels,
)

# Knowledge-based prompt: presents clinical criteria as context, not examples
KB_TEMPLATE = """Using the clinical criteria below, classify the following text as one of: {labels}.

Clinical criteria for suicide risk assessment:
{context}

Based on these criteria, does the text indicate suicide risk?
Text: {text}
Label:"""


class PromptBuilder:
    """Builds prompts for suicide classification."""

    def __init__(self):
        self.labels = get_labels()
        self.labels_str = ", ".join(self.labels)

    def build_zero_shot(self, text: str) -> str:
        return ZERO_SHOT_TEMPLATE.format(
            labels=self.labels_str,
            text=text,
        )

    def build_few_shot(
        self,
        text: str,
        examples: List[Tuple[str, str]],
    ) -> str:
        formatted_examples = "\n\n".join(
            EXAMPLE_TEMPLATE.format(text=ex_text, label=ex_label)
            for ex_text, ex_label in examples
        )
        return FEW_SHOT_TEMPLATE.format(
            labels=self.labels_str,
            examples=formatted_examples,
            text=text,
        )

    def build_knowledge_based(
        self,
        text: str,
        context_chunks: List[Tuple[str, str]],
    ) -> str:
        """
        Build a prompt with clinical knowledge as context.
        Chunks are (text, section_name) from the knowledge base.
        """
        formatted_context = "\n\n".join(
            f"[{section}]: {chunk_text}"
            for chunk_text, section in context_chunks
        )
        return KB_TEMPLATE.format(
            labels=self.labels_str,
            context=formatted_context,
            text=text,
        )

    def build(
        self,
        text: str,
        examples: List[Tuple[str, str]] | None = None,
        is_knowledge_base: bool = False,
    ) -> str:
        if examples:
            if is_knowledge_base:
                return self.build_knowledge_based(text, examples)
            return self.build_few_shot(text, examples)
        return self.build_zero_shot(text)