"""Command line interface.

persona-pipeline run --persona personas/example.yaml --topic "morning routines"
persona-pipeline backends
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

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


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check that every backend the persona file asks for is ready to use."""
    import os
    import urllib.request

    config = load_config(args.persona)
    choice = config.pipeline.backends
    problems = 0

    def report(ok: bool, what: str, hint: str = "") -> None:
        nonlocal problems
        problems += not ok
        print(f"  [{'ok' if ok else '!!'}] {what}" + ("" if ok else f"\n       {hint}"))

    print(
        f"Persona: {config.persona.name}  (text={choice.text}, image={choice.image}, "
        f"vision={choice.vision})"
    )
    if "gemini" in (choice.text, choice.image, choice.vision):
        has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        report(has_key, "Gemini API key found", "Add GEMINI_API_KEY=... to .env in this folder.")
        try:
            import google.genai  # noqa: F401

            report(True, "Gemini library installed")
        except ImportError:
            report(False, "Gemini library installed", 'Run: pip install -e ".[gemini]"')
    if choice.image == "drawthings":
        url = (os.environ.get("DRAWTHINGS_URL") or config.pipeline.drawthings.url).rstrip("/")
        try:
            with urllib.request.urlopen(f"{url}/sdapi/v1/options", timeout=5):
                pass
            report(True, f"Draw Things answering at {url}")
        except Exception:
            report(
                False,
                f"Draw Things answering at {url}",
                "Open Draw Things, then Settings > API Server > turn it on.",
            )
    print("All good." if not problems else f"{problems} thing(s) to fix.")
    return 0 if not problems else 1


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

    doc = sub.add_parser("doctor", help="check keys and apps needed by a persona file")
    doc.add_argument("--persona", required=True, help="path to a persona YAML file")
    doc.set_defaults(func=_cmd_doctor)

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
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: file not found: {exc.filename}", file=sys.stderr)
    except (yaml.YAMLError, ValidationError) as exc:
        print(f"Error: the persona file is not valid.\n{exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
