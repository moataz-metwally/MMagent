"""CLI entry point: python -m MMagent "question" --depth quick"""

from __future__ import annotations

import argparse
import asyncio

from .agent import BOLD, DIM, RESET, create_agent


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMagent — internet-grounded research agent",
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Your research question (one or more words)",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Search depth: quick (fast answer), standard (balanced), deep (exhaustive)",
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    print(f"{BOLD}Question:{RESET} {question}  {DIM}[depth: {args.depth}]{RESET}\n")

    agent = create_agent(depth=args.depth)
    answer = await agent.prompt(question)

    print(f"\n{'─' * 60}")
    if answer:
        print(f"\n{BOLD}Answer:{RESET}\n{answer}")
    else:
        print(f"\n{BOLD}Agent finished.{RESET} (no text in final message)")


if __name__ == "__main__":
    asyncio.run(main())
