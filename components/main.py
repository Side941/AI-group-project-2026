"""
main.py
=======
Single entrypoint for the ICD-11 knowledge-base pipeline.

Runs:
1) chunking  (PDF -> JSON chunks)
2) ingestion (JSON chunks -> ChromaDB vectors)
"""

from __future__ import annotations

import argparse

try:
    from config import (
        BATCH_SIZE,
        CHROMA_PATH,
        CHUNKS_PATH,
        COLLECTION_NAME,
        CONTENT_END_PAGE,
        CONTENT_START_PAGE,
        EMBEDDING_MODEL,
        PDF_PATH,
    )
    from chunker import CHUNK_WORD_OVERLAP, MAX_CHUNK_WORDS, run_chunking
    from ingestion import run_ingestion
except ModuleNotFoundError:
    from components.config import (
        BATCH_SIZE,
        CHROMA_PATH,
        CHUNKS_PATH,
        COLLECTION_NAME,
        CONTENT_END_PAGE,
        CONTENT_START_PAGE,
        EMBEDDING_MODEL,
        PDF_PATH,
    )
    from components.chunker import CHUNK_WORD_OVERLAP, MAX_CHUNK_WORDS, run_chunking
    from components.ingestion import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chunking then ingestion for the ICD-11 knowledge base.",
    )
    parser.add_argument("--pdf-path", type=str, default=str(PDF_PATH))
    parser.add_argument("--chunks-path", type=str, default=str(CHUNKS_PATH))
    parser.add_argument("--chroma-path", type=str, default=str(CHROMA_PATH))
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME)
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--start-page", type=int, default=CONTENT_START_PAGE)
    parser.add_argument("--end-page", type=int, default=CONTENT_END_PAGE)
    parser.add_argument("--max-words", type=int, default=MAX_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=CHUNK_WORD_OVERLAP)
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Use existing chunks JSON and only run ingestion.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing Chroma collection and re-ingest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.skip_chunking:
        print("\n=== Step 1/2: Chunking ===")
        run_chunking(
            pdf_path=args.pdf_path,
            chunks_path=args.chunks_path,
            start_page=args.start_page,
            end_page=args.end_page,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
    else:
        print("\n=== Step 1/2: Chunking skipped ===")

    print("\n=== Step 2/2: Ingestion ===")
    run_ingestion(
        chunks_path=args.chunks_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
