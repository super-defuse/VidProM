#!/usr/bin/env python3
"""Extract common nouns from English prompts with spaCy.

The implementation mirrors ``Chai._get_noun_from_prompt`` in CHAI:
spaCy's ``en_core_web_sm`` model assigns a part-of-speech tag to each token,
and only tokens tagged ``NOUN`` are returned. Token order and spelling are
preserved, and repeated nouns are kept.

Install the runtime dependency and model before using this script::

    pip install spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import spacy
from spacy.language import Language


def _get_noun_from_prompt(prompt: str, nlp: Language) -> list[str]:
    """Return every token whose spaCy part-of-speech is ``NOUN``.

    This intentionally matches CHAI's private helper. It does not remove
    duplicates, normalize case, or include proper nouns (``PROPN``).
    """

    return [token.text for token in nlp(prompt) if token.pos_ == "NOUN"]


def clean_prompt(prompt: str) -> str:
    """Apply the cleaning performed by CHAI before noun extraction."""

    return prompt.split("aesthetic score")[0]


def extract_prompts(
    prompts: Iterable[str], nlp: Language, *, clean: bool = False
) -> list[list[str]]:
    """Extract nouns from multiple prompts, preserving input order."""

    if clean:
        prompts = (clean_prompt(prompt) for prompt in prompts)
    return [_get_noun_from_prompt(prompt, nlp) for prompt in prompts]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract spaCy NOUN tokens from English prompts."
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="Prompt text. If omitted, read one prompt per line from stdin.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Read one prompt per line from this UTF-8 text file.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model to load (default: en_core_web_sm).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help='Strip text starting at "aesthetic score", as CHAI does.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.file and args.prompts:
        raise SystemExit("Use either positional prompts or --file, not both.")

    if args.file:
        prompts = args.file.read_text(encoding="utf-8").splitlines()
    elif args.prompts:
        prompts = [" ".join(args.prompts)]
    else:
        prompts = sys.stdin.read().splitlines()

    try:
        nlp = spacy.load(args.model)
    except OSError as error:
        raise SystemExit(
            f"Could not load spaCy model {args.model!r}. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from error

    for nouns in extract_prompts(prompts, nlp, clean=args.clean):
        print("\t".join(nouns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
