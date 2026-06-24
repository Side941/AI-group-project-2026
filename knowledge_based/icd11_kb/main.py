"""
main.py
=======
Entry point for the ICD-11 knowledge base pipeline.

Runs all three stages end-to-end:
  1. Chunking  — PDF → structured JSON chunks
  2. Ingestion — JSON chunks → ChromaDB vectors
  3. Retrieval — sanity-check queries against the populated collection

Usage
-----
    python main.py

Or import individual stages directly:
    from chunker   import run_chunking
    from ingestion import run_ingestion
    from retrieval import initialise_retrieval, query_icd11, get_icd11_context
"""

from chunker   import run_chunking
from ingestion import run_ingestion
from retrieval import query_icd11, get_icd11_context


def main() -> None:
    # ── Step 1: chunk the PDF ─────────────────────────────────────────────────
    run_chunking()

    # ── Step 2: embed and store in ChromaDB ───────────────────────────────────
    run_ingestion()

    # ── Step 3: sanity-check queries ─────────────────────────────────────────
    print("\n── Test 1: general retrieval ─────────────────────────────────────")
    query_icd11("I don't enjoy anything anymore and I feel completely worthless")

    print("\n── Test 2: Essential Features filter ─────────────────────────────")
    query_icd11(
        "I don't want to wake up tomorrow",
        section_filter="Essential Features",
    )

    print("\n── Test 3: Boundary with Normality ───────────────────────────────")
    query_icd11(
        "I've been feeling really sad since my dog died last week",
        section_filter="Boundary with Normality",
    )

    print("\n── Test 4: formatted LLM context block ───────────────────────────")
    context = get_icd11_context(
        "I feel like everyone would be better off without me",
        n_results=3,
    )
    print(context)

    print("\n── Test 5: full RAG prompt example ───────────────────────────────")
    post    = "I've been crying every day for two weeks and I can't get out of bed"
    context = get_icd11_context(post, n_results=5)

    prompt = (
        "You are a clinical assistant helping assess mental health risk.\n"
        "Use the ICD-11 clinical reference below to inform your response.\n\n"
        "--- ICD-11 Clinical Reference ---\n"
        f"{context}\n"
        "---------------------------------\n\n"
        f"Post: {post}\n\n"
        "Based on the clinical reference, what disorder categories are most "
        "relevant to this post? Explain your reasoning."
    )
    print(prompt)


if __name__ == "__main__":
    main()
