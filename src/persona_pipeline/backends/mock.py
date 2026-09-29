"""Offline mock backends.

They need no API key and no network, return deterministic results, and produce
real image files. This lets anyone clone the repo and run the full pipeline in a
few seconds, and lets the test suite exercise the orchestration logic.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from persona_pipeline.models import GeneratedImage, ImagePrompt

ASPECT_SIZES = {
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "9:16": (1080, 1920),
}


def _digest(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _field(text: str, label: str, default: str = "") -> str:
    match = re.search(rf"^{label}:\s*(.+)$", text, flags=re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else default


class _MockBase:
    name = "mock"

    def __init__(self, settings=None):
        self.settings = settings


class MockText(_MockBase):
    def complete_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict:
        topic = _field(prompt, "Topic", "an idea")
        subject = _field(prompt, "Subject", "a creator")
        signature = _field(prompt, "Signature", "")
        avoid = _field(prompt, "Avoid", "")

        if task == "write_prompt":
            feedback = _field(prompt, "Fix", "")
            extra = f", {feedback}" if feedback else ""
            return {
                "positive": (
                    f"{subject}, visual story about {topic}, {signature}{extra}, "
                    "editorial photo, highly detailed"
                ).replace(", ,", ","),
                "negative": avoid or "blurry, extra fingers, text, watermark",
            }

        if task == "write_caption":
            words = [w for w in re.findall(r"[A-Za-z]+", topic.lower()) if len(w) > 3]
            return {
                "text": f"A little note on {topic}. What would you add? (mock caption)",
                "hashtags": words[:5],
                "alt_text": f"{subject} in a scene about {topic}",
            }

        raise ValueError(f"MockText has no canned answer for task '{task}'")


class MockImage(_MockBase):
    def generate(self, prompt: ImagePrompt, out_path: Path) -> GeneratedImage:
        width, height = ASPECT_SIZES.get(prompt.aspect_ratio, ASPECT_SIZES["4:5"])
        seed = _digest(prompt.positive, str(prompt.seed))
        top = ((seed >> 0) & 0xFF, (seed >> 8) & 0xFF, (seed >> 16) & 0xFF)
        bottom = ((seed >> 24) & 0xFF, (seed >> 32) & 0xFF, (seed >> 40) & 0xFF)

        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        for y in range(height):
            t = y / max(height - 1, 1)
            color = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
            draw.line([(0, y), (width, y)], fill=color)

        font = ImageFont.load_default(size=36)
        body = textwrap.fill(prompt.positive, width=48)
        draw.multiline_text((60, 60), "MOCK IMAGE", font=font, fill="white")
        draw.multiline_text((60, 130), body, font=font, fill="white", spacing=10)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return GeneratedImage(path=out_path, width=width, height=height, backend=self.name)


class MockVision(_MockBase):
    def inspect_json(
        self, task: str, image_path: Path, instruction: str, schema: dict[str, Any]
    ) -> dict:
        if task != "critique":
            raise ValueError(f"MockVision has no canned answer for task '{task}'")
        seed = _digest(hashlib.sha256(Path(image_path).read_bytes()).hexdigest())
        score = round(6.0 + (seed % 36) / 10, 1)  # 6.0 to 9.5
        notes = ["composition reads clearly at phone size"]
        if score < 7.5:
            notes.append("signature element is not prominent enough")
        return {"score": score, "notes": notes}
