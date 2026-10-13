#!/usr/bin/env python3
"""Build the edge-case and image request fixtures for the Clef-flash oracle.

Deterministic: the same inputs produce the same files, so re-running is a no-op diff.
Edge records target one encoder/head path each (see EDGE below). Image records come in
pairs: the image record and a text-only twin with the same state and questions, so the
vision tower's contribution is measurable as the difference between the two.

Token-length targets are set in characters here (Qwen tokenizers run ~3.5-4 chars/token on
English); the pod's `oracle.py --dry-run` reports the real `input_tokens`, and the plan's
acceptance ranges are checked there, not here.

Usage:
    python scripts/make_fixtures.py   # writes fixtures/requests/{edge,images}.jsonl + images/
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "fixtures" / "requests"
IMG = REQ / "images"

# The model card's example record: three question types in one request.
CARD_TRIO = {
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}

URGENT_Q = "Does any item in the state describe breaking, time-critical news?"

TICKET = (
    "Hi, I was charged twice for my October invoice (INV-20419). The second charge posted this "
    "morning. I'd like the duplicate refunded. Not urgent, but please confirm by Friday."
)


def pulse_states(n: int) -> list[str]:
    path = REQ / "pulse50.jsonl"
    if not path.exists():
        raise SystemExit("run scripts/export_pulse.py first (long records reuse Pulse states)")
    states = [
        json.loads(line)["request"]["state"] for line in path.read_text().splitlines() if line
    ]
    return states[:n]


def long_state(target_chars: int) -> str:
    states = pulse_states(50)
    out, i = [], 0
    while sum(len(s) + 2 for s in out) < target_chars:
        out.append(states[i % len(states)])
        i += 1
    return "\n\n".join(out)


def edge_records() -> list[dict]:
    recs: list[dict] = []

    def add(rid: str, request: dict, targets: str):
        recs.append({"id": rid, "source": {"kind": "edge", "targets": targets}, "request": request})

    add(
        "card_trio",
        CARD_TRIO,
        "multi-question record; stage-2 cross-field attention; all three types",
    )

    add(
        "no_instruction",
        {
            "state": TICKET,
            "questions": {"refund_eligible": {"type": "score", "criteria": ["no", "maybe", "yes"]}},
        },
        "instructions omitted → question span is the id token(s)",
    )

    add(
        "unsorted_choice",
        {
            "state": TICKET,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "technical": "Outages, errors, configuration",
                        "billing": "Payments, invoices, refunds",
                        "sales": "Plans and upgrades",
                    },
                }
            },
        },
        "choice ids given out of order → scored sorted, returned in request order",
    )

    add(
        "noul_custom",
        {
            "state": TICKET,
            "questions": {
                "wants_refund": {
                    "type": "noul",
                    "instructions": "Does the customer ask for money back?",
                    "criteria": {
                        "true": "They ask for a refund or reversal.",
                        "false": "They do not.",
                    },
                }
            },
        },
        "noul with overridden true/false descriptions",
    )

    add(
        "json_state",
        {
            "state": {
                "invoice": {
                    "vendor": "Acme",
                    "total": 1250.0,
                    "currency": "USD",
                    "status": "overdue",
                },
                "zeta": 1,
                "alpha": [3, 1, 2],
            },
            "questions": {
                "status": {
                    "type": "choice",
                    "instructions": "What is the invoice status?",
                    "criteria": {
                        "paid": "Invoice is paid.",
                        "overdue": "Invoice is past due.",
                        "draft": "Not sent.",
                    },
                },
                "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
            },
        },
        "dict state → render() with sort_keys and compact separators",
    )

    options = {f"opt_{i:02d}": f"Category number {i}: {w}" for i, w in enumerate(WORDS40)}
    add(
        "many_options",
        {
            "state": "The user wrote: my package arrived crushed and the contents are broken.",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Which category fits the message?",
                    "criteria": options,
                }
            },
        },
        "40-option choice → head padding path (O > 16), option-set attention",
    )

    add(
        "long_2k",
        {
            "state": long_state(9000),
            "questions": {
                "any_urgent": {
                    "type": "noul",
                    "instructions": URGENT_Q,
                },
                "count_bucket": {
                    "type": "score",
                    "instructions": "Roughly how many distinct news items are in the state?",
                    "criteria": ["fewer than 5", "5 to 15", "16 to 30", "more than 30"],
                },
            },
        },
        "~2k-token state; attention layers at moderate length",
    )

    add(
        "trunc_16k",
        {
            "state": long_state(80000),
            "questions": {
                "any_urgent": {
                    "type": "noul",
                    "instructions": URGENT_Q,
                }
            },
        },
        "state > 16,384 tokens → encode_record truncates the state; input_tokens == 16384",
    )
    return recs


WORDS40 = [
    "damaged delivery",
    "late delivery",
    "missing item",
    "wrong item",
    "refund request",
    "billing error",
    "duplicate charge",
    "cancel subscription",
    "upgrade plan",
    "downgrade plan",
    "login problem",
    "password reset",
    "account locked",
    "data export",
    "privacy request",
    "bug report",
    "feature request",
    "api outage",
    "slow performance",
    "mobile app crash",
    "shipping address",
    "gift card",
    "promo code",
    "price match",
    "warranty claim",
    "return label",
    "exchange item",
    "store hours",
    "job application",
    "press inquiry",
    "partnership",
    "legal notice",
    "spam",
    "compliment",
    "complaint about staff",
    "accessibility",
    "language support",
    "unsubscribe",
    "wholesale order",
    "other",
]


def make_images() -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont

    IMG.mkdir(parents=True, exist_ok=True)
    mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)

    # 1. receipt: legible total
    im = Image.new("RGB", (480, 640), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "NORTHSIDE HARDWARE", font=bold, fill="black")
    y = 90
    for name, price in [
        ("Paint roller", "8.49"),
        ("Wood screws 100ct", "6.99"),
        ("Drop cloth", "12.00"),
    ]:
        d.text((40, y), f"{name:<22}{price:>8}", font=mono, fill="black")
        y += 36
    d.line((40, y + 6, 440, y + 6), fill="black", width=2)
    d.text((40, y + 20), f"{'TOTAL':<22}{'27.48':>8}", font=bold, fill="black")
    d.text((40, y + 70), "VISA ****4421   10/11/2026", font=mono, fill="black")
    im.save(IMG / "receipt.png")

    # 2. chart: a bar chart whose tallest bar is unambiguous
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.bar(["Q1", "Q2", "Q3", "Q4"], [12, 19, 31, 22], color="#4c56c9")
    ax.set_title("Revenue by quarter ($M)")
    ax.set_ylim(0, 35)
    fig.tight_layout()
    fig.savefig(IMG / "chart.png")
    plt.close(fig)

    # 3. no text: colored shapes
    im = Image.new("RGB", (512, 512), (245, 240, 230))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 260, 260), fill=(220, 50, 50))
    d.rectangle((280, 280, 460, 460), fill=(40, 90, 200))
    d.polygon([(300, 60), (460, 60), (380, 220)], fill=(40, 160, 90))
    im.save(IMG / "shapes.png")

    recs = []

    def pair(rid, image, state, questions, targets):
        recs.append(
            {
                "id": f"img_{rid}",
                "source": {"kind": "image", "targets": targets},
                "request": {"state": state, "questions": questions},
                "images": [f"fixtures/requests/images/{image}"],
            }
        )
        recs.append(
            {
                "id": f"img_{rid}_textonly",
                "source": {"kind": "image-twin", "targets": "same request without the image"},
                "request": {"state": state, "questions": questions},
            }
        )

    pair(
        "receipt",
        "receipt.png",
        {"task": "Review the attached receipt."},
        {
            "legible": {"type": "noul", "instructions": "Is the receipt total legible?"},
            "total_bucket": {
                "type": "choice",
                "instructions": "Which range contains the total?",
                "criteria": {
                    "under_10": "Less than $10",
                    "10_to_50": "$10 to $50",
                    "over_50": "More than $50",
                },
            },
        },
        "vision: OCR-like read of a rendered total",
    )
    pair(
        "chart",
        "chart.png",
        {"task": "Read the attached chart."},
        {
            "best_quarter": {
                "type": "choice",
                "instructions": "Which quarter has the highest bar?",
                "criteria": {"Q1": None, "Q2": None, "Q3": None, "Q4": None},
            }
        },
        "vision: relative magnitude in a chart",
    )
    pair(
        "shapes",
        "shapes.png",
        {"task": "Describe the attached picture."},
        {
            "has_text": {
                "type": "noul",
                "instructions": "Does the image contain any written text?",
            },
            "red_shape": {
                "type": "choice",
                "instructions": "What shape is the red object?",
                "criteria": {"circle": None, "square": None, "triangle": None},
            },
        },
        "vision: no text, pure shape/color",
    )
    return recs


def write_jsonl(path: Path, recs: list[dict]) -> None:
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    edge = edge_records()
    write_jsonl(REQ / "edge.jsonl", edge)
    imgs = make_images()
    write_jsonl(REQ / "images.jsonl", imgs)
    for r in edge + imgs:
        chars = len(json.dumps(r["request"]["state"]))
        print(f"{r['id']:<22} state_chars={chars:>6}  {r['source']['targets']}")


if __name__ == "__main__":
    main()
