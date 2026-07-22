"""
smoke_paths.py
==============
Quick existence check for assets the current pipeline needs.

    python experiments/smoke_paths.py
"""

from __future__ import annotations

from _common import bootstrap

bootstrap()

from components.config import (  # noqa: E402
    CHROMA_PATH,
    CHUNKS_PATH,
    FEWSHOT_CHROMA_PATH,
    FEWSHOT_MULTICLASS_EXAMPLES_PATH,
    MULTICLASS_DEV_PATH,
    MULTICLASS_EVAL_PATH,
    PDF_PATH,
)


def check(label: str, path, *, required: bool = True) -> bool:
    ok = path.exists()
    flag = "OK" if ok else ("MISSING" if required else "optional-missing")
    print(f"  [{flag:16}] {label}")
    print(f"                   {path}")
    return ok or not required


def main() -> int:
    print("=== Pipeline asset check ===\n")

    required_ok = True
    print("ICD-11 knowledge base")
    required_ok &= check("PDF (chunking source)", PDF_PATH, required=False)
    required_ok &= check("Chunks JSON", CHUNKS_PATH)
    required_ok &= check("ChromaDB store", CHROMA_PATH)

    print("\nProcessed datasets")
    required_ok &= check("Multiclass eval", MULTICLASS_EVAL_PATH)
    required_ok &= check("Multiclass dev", MULTICLASS_DEV_PATH)

    print("\nRaw / few-shot (optional for retrieval smokes)")
    check("Few-shot multiclass JSON", FEWSHOT_MULTICLASS_EXAMPLES_PATH, required=False)
    check("Few-shot ChromaDB", FEWSHOT_CHROMA_PATH, required=False)

    print()
    if required_ok:
        print("PASS — required pipeline assets are present.")
        return 0

    print("FAIL — fix missing required assets before running retrieval/RAG smokes.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
