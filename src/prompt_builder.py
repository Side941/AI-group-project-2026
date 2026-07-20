"""
prompt_builder.py
=================
Builds prompts for suicide classification.
Supports: zero-shot, KB only, Reddit examples only, and KB + Reddit combined.
"""

from __future__ import annotations

from typing import List, Tuple
from src.config import ZERO_SHOT_TEMPLATE, KB_TEMPLATE, get_labels

# Combined prompt template
COMBINED_TEMPLATE = """Using the clinical criteria below AND the example posts, classify the following text as one of: {labels}.

Clinical criteria for suicide risk assessment:
{context}

Example posts:
{examples}

Based on the clinical criteria and the patterns in the examples, classify this text:
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

    def build_knowledge_based(
        self,
        text: str,
        context_chunks: List[Tuple[str, str]],
    ) -> str:
        """Build prompt with KB chunks only."""
        formatted_context = "\n\n".join(
            f"[{section}]: {chunk_text}"
            for chunk_text, section in context_chunks
        )
        return KB_TEMPLATE.format(
            labels=self.labels_str,
            context=formatted_context,
            text=text,
        )

    def build_few_shot(
        self,
        text: str,
        examples: List[Tuple[str, str]],
    ) -> str:
        """Build prompt with Reddit examples only."""
        formatted_examples = "\n\n".join(
            f"Text: {ex_text}\nLabel: {ex_label}"
            for ex_text, ex_label in examples
        )
        return FEW_SHOT_TEMPLATE.format(
            labels=self.labels_str,
            examples=formatted_examples,
            text=text,
        )

    def build_combined(
        self,
        text: str,
        kb_chunks: List[Tuple[str, str]],
        reddit_examples: List[Tuple[str, str]],
    ) -> str:
        """Build prompt with BOTH KB chunks AND Reddit examples."""
        # Format KB chunks
        formatted_context = "\n\n".join(
            f"[{section}]: {chunk_text}"
            for chunk_text, section in kb_chunks
        )
        
        # Format Reddit examples
        formatted_examples = "\n\n".join(
            f"Text: {ex_text}\nLabel: {ex_label}"
            for ex_text, ex_label in reddit_examples
        )
        
        return COMBINED_TEMPLATE.format(
            labels=self.labels_str,
            context=formatted_context,
            examples=formatted_examples,
            text=text,
        )

    def build(
        self,
        text: str,
        kb_chunks: List[Tuple[str, str]] | None = None,
        reddit_examples: List[Tuple[str, str]] | None = None,
        use_kb: bool = False,
        use_examples: bool = False,
    ) -> str:
        """Main dispatch method."""
        # Combined (KB + Reddit examples)
        if kb_chunks and reddit_examples:
            return self.build_combined(text, kb_chunks, reddit_examples)
        
        # KB only
        if kb_chunks and use_kb:
            return self.build_knowledge_based(text, kb_chunks)
        
        # Reddit examples only
        if reddit_examples and use_examples:
            return self.build_few_shot(text, reddit_examples)
        
        # Zero-shot (default)
        return self.build_zero_shot(text)