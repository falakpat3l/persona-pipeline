"""Command line interface.

persona-pipeline run --persona personas/example.yaml --topic "morning routines"
persona-pipeline backends
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from persona_pipeline import __version__, backends
from persona_pipeline.config import load_config
from persona_pipeline.env import load_dotenv
from persona_pipeline.models import Brief
from persona_pipeline.pipeline import Pipeline


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.persona)
    if args.text:
        config.pipeline.backends.text = args.text
    if args.image:
        config.pipeline.backends.image = args.image
    if args.vision:
        config.pipeline.backends.vision = args.vision

    brief = Brief(topic=args.topic, angle=args.angle, format=args.format)
    job = Pipeline(config).run(brief, output_root=Path(args.out) if args.out else None)

    print()
    print(f"Job:     {job.job_id}")
    print(f"Status:  {'OK' if job.succeeded else 'FAILED'}")
    if job.critique:
        print(f"Score:   {job.critique.score:.1f} (best of {len(job.attempts)} attempts)")
    print(f"Output:  {job.output_dir}")
    if job.caption:
        print()
        print(job.caption.render())
    return 0 if job.succeeded else 1


def _cmd_backends(_: argparse.Namespace) -> int:
    for role in ("text", "image", "vision"):
        print(f"{role:7} {', '.join(backends.available(role))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="persona-pipeline", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate one post from a topic")
    run.add_argument("--persona", required=True, help="path to a persona YAML file")
    run.add_argument("--topic", required=True, help="what the post is about")
    run.add_argument("--angle", help="optional hook or point of view")
    run.add_argument("--format", default="portrait", choices=["portrait", "square", "story"])
    run.add_argument("--out", help="output folder (default from config)")
    run.add_argument("--text", help="override text backend")
    run.add_argument("--image", help="override image backend")
    run.add_argument("--vision", help="override vision backend")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("backends", help="list available backends")
    lst.set_defaults(func=_cmd_backends)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()  # picks up GEMINI_API_KEY etc. from ./.env if present
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if not args.verbose:
        # Keep the output to our own stage log; SDK and HTTP chatter only with -v.
        for noisy in ("httpx", "httpcore", "google_genai", "google.genai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
