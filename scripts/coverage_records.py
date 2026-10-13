"""Coverage records: hosted-API limits, encoder corner cases, and customer-shaped workloads.

Imported by scripts/make_fixtures.py. Everything here is synthetic. Each record names the
path or workload it targets so a parity failure on it points at one place.

Hosted limits (Workers AI model page for @cf/cloudflare/clef): 1-64 questions, 2-255 choice
options, 2-10 score levels. encode_record's default max_length is 16,384 tokens and the
schema must fit before any state is admitted.
"""

from __future__ import annotations

TICKET = (
    "Subject: card declined at checkout\n\n"
    "Tried to pay for my order three times tonight and the card keeps getting declined, "
    "but my bank says nothing is wrong on their end. Order #A81-2231. I need this shipped "
    "before Thursday for a gift. Pretty frustrated at this point."
)

# BANKING77-shaped intent set: 77 banking intents in the dataset's naming style.
INTENTS77 = """activate_my_card age_limit apple_pay_or_google_pay atm_support automatic_top_up
balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit
beneficiary_not_allowed cancel_transfer card_about_to_expire card_acceptance card_arrival
card_delivery_estimate card_linking card_not_working card_payment_fee_charged
card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed cash_withdrawal_charge
cash_withdrawal_not_recognised change_pin compromised_card contactless_not_working
country_support declined_card_payment declined_cash_withdrawal declined_transfer
direct_debit_payment_not_recognised disposable_card_limits edit_personal_details exchange_charge
exchange_rate exchange_via_app extra_charge_on_statement failed_transfer fiat_currency_support
get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card
lost_or_stolen_card lost_or_stolen_phone order_physical_card passcode_forgotten
pending_card_payment pending_cash_withdrawal pending_top_up pending_transfer pin_blocked
receiving_money refund_not_showing_up request_refund reverted_card_payment
supported_cards_and_currencies terminate_account top_up_by_bank_transfer_charge
top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted
topping_up_by_card transaction_charged_twice transfer_fee_charged transfer_into_account
transfer_not_received_by_recipient transfer_timing unable_to_verify_identity verify_my_identity
verify_source_of_funds verify_top_up virtual_card_not_working visa_or_mastercard
why_verify_identity wrong_amount_of_cash_received wrong_exchange_rate_for_cash_withdrawal""".split()
assert len(INTENTS77) == 77, len(INTENTS77)

TOOLS = {
    "get_order_status": "Look up an order's shipping and payment status by order id.",
    "retry_payment": "Retry the most recent failed payment on an order.",
    "update_shipping_address": "Change the delivery address on an open order.",
    "issue_refund": "Refund a captured payment in full or in part.",
    "escalate_to_human": "Hand the conversation to a support agent.",
    "none": "No tool is needed; answer from the conversation alone.",
}

INVOICE = {
    "invoice_id": "INV-20419",
    "vendor": {"name": "Acme Fasteners", "country": "DE"},
    "currency": "EUR",
    "total": 1250.0,
    "issued": "2026-08-30",
    "due": "2026-09-29",
    "paid": False,
    "lines": [
        {"sku": "M6-40", "qty": 500, "unit_price": 1.2},
        {"sku": "M8-60", "qty": 400, "unit_price": 1.625},
    ],
}

RAG_QUERY = "What is the maximum number of questions a Clef request can contain?"
RAG_PASSAGES = {
    "1": (
        "Clef accepts 1 to 64 questions per request, 2 to 255 options for a choice question "
        "and 2 to 10 levels for a score question. Output tokens are not billed."
    ),
    "2": (
        "Clef is post-trained from Qwen3.8-27B and keeps the vision encoder, so images and "
        "video frames can be passed alongside the state."
    ),
    "3": (
        "A treble clef fixes the G above middle C on the second line of the staff; the bass "
        "clef fixes the F below middle C on the fourth line."
    ),
}

COMMENT = (
    "lol this update is garbage. whoever shipped it should be fired, and i mean that. "
    "you people clearly dont test anything. uninstalling."
)


def records() -> list[dict]:
    recs: list[dict] = []

    def add(rid: str, request: dict, targets: str, kind: str = "coverage", **extra):
        rec = {"id": rid, "source": {"kind": kind, "targets": targets}, "request": request}
        rec.update(extra)
        recs.append(rec)

    # ---- hosted limits ----------------------------------------------------------------
    add(
        "intent77",
        {
            "state": TICKET,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking intent best matches the message?",
                    "criteria": {k: None for k in INTENTS77},
                }
            },
        },
        "77-option choice (BANKING77 shape, Clef's headline benchmark)",
    )
    add(
        "opts255",
        {
            "state": "The customer asked for the part that goes between the pump and the hose.",
            "questions": {
                "sku": {
                    "type": "choice",
                    "instructions": "Which catalogue entry does the customer mean?",
                    "criteria": {
                        f"sku_{i:03d}": f"Part {i}: {w}" for i, w in enumerate(_catalogue(255))
                    },
                }
            },
        },
        "255 options, the hosted choice cap; head attention over a large option set",
    )
    add(
        "q64",
        {
            "state": TICKET,
            "questions": {
                f"q{i:02d}": {
                    "type": "noul",
                    "instructions": _NOUL_QUESTIONS[i % len(_NOUL_QUESTIONS)],
                }
                for i in range(64)
            },
        },
        "64 questions, the hosted cap; stage-2 self-attention over 64 fields",
    )
    add(
        "score10",
        {
            "state": TICKET,
            "questions": {
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer, 0 (calm) to 9 (furious)?",
                    "criteria": [f"level {i}" for i in range(10)],
                }
            },
        },
        "10 score levels, the hosted cap; expected-value decoder over a wide scale",
    )

    # ---- encoder corner cases -------------------------------------------------------
    add(
        "empty_instructions",
        {
            "state": TICKET,
            "questions": {
                "needs_shipping_help": {
                    "type": "noul",
                    "instructions": "",
                }
            },
        },
        'instructions "" is treated like missing: span becomes the question id',
    )
    add(
        "unicode_options",
        {
            "state": "Kunde schreibt: Die Lieferung kam beschädigt an, bitte um Ersatz. Danke! 🙏",
            "questions": {
                "sprache": {
                    "type": "choice",
                    "instructions": "In welcher Sprache ist die Nachricht?",
                    "criteria": {
                        "de": "Deutsch",
                        "fr": "Français",
                        "ja": "日本語",
                        "en": "English",
                    },
                },
                "stimmung": {
                    "type": "choice",
                    "instructions": "Ton der Nachricht?",
                    "criteria": {
                        "freundlich": "höflich 🙂",
                        "neutral": "sachlich",
                        "wuetend": "wütend 😡",
                    },
                },
            },
        },
        "non-ASCII in state, ids and descriptions; render() uses ensure_ascii=False",
    )
    add(
        "list_state",
        {
            "state": [
                {"t": "09:00", "event": "deploy v2.14 started"},
                {"t": "09:04", "event": "error rate 0.2% → 3.1%"},
                {"t": "09:06", "event": "rollback initiated"},
                {"t": "09:09", "event": "error rate back to 0.2%"},
            ],
            "questions": {
                "incident": {"type": "noul", "instructions": "Did an incident occur?"},
                "resolved": {
                    "type": "noul",
                    "instructions": "Is it resolved by the end of the log?",
                },
                "cause": {
                    "type": "choice",
                    "instructions": "Most likely cause?",
                    "criteria": {
                        "deploy": "A code deploy",
                        "traffic": "A traffic spike",
                        "infra": "Infrastructure",
                    },
                },
            },
        },
        "state is a JSON list; render() path for non-dict, non-str",
    )
    add(
        "schema_overflow",
        {
            "state": "x",
            "questions": {
                f"q{i:02d}": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "criteria": {f"o{j:03d}": f"Option {j} of question {i}" for j in range(255)},
                }
                for i in range(64)
            },
        },
        "64 x 255 options: schema alone exceeds max_length; encode_record raises, worker 422s",
        expect_error="schema requires",
    )

    # ---- customer-shaped workloads --------------------------------------------------
    add(
        "triage_support",
        {
            "state": TICKET,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should own this?",
                    "criteria": {
                        "payments": "Card and checkout failures",
                        "shipping": "Delivery and address changes",
                        "account": "Login, identity, profile",
                        "other": "None of the above",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "Customer impact right now",
                    "criteria": ["none", "minor", "major", "blocking"],
                },
                "sentiment": {
                    "type": "choice",
                    "instructions": "Tone of the message",
                    "criteria": {"positive": None, "neutral": None, "negative": None},
                },
                "deadline": {"type": "noul", "instructions": "Does the customer state a deadline?"},
            },
        },
        "support triage: route + severity + sentiment + deadline, with an `other` escape",
        kind="workload",
    )
    add(
        "tool_select",
        {
            "state": {
                "conversation": [
                    {
                        "role": "user",
                        "content": "my payment failed twice, can you just try it again?",
                    },
                ],
                "tools": TOOLS,
            },
            "questions": {
                "tool": {
                    "type": "choice",
                    "instructions": "Which tool should the agent call next?",
                    "criteria": TOOLS,
                },
                "needs_clarification": {
                    "type": "noul",
                    "instructions": "Is information missing that the agent must ask for first?",
                },
            },
        },
        "tool selection (When2Call shape): choice over tool names + a none option",
        kind="workload",
    )
    add(
        "invoice_extract",
        {
            "state": INVOICE,
            "questions": {
                "overdue": {
                    "type": "noul",
                    "instructions": "Is the invoice past due as of 2026-10-13?",
                },
                "currency": {
                    "type": "choice",
                    "instructions": "Invoice currency",
                    "criteria": {"USD": None, "EUR": None, "GBP": None, "JPY": None},
                },
                "amount_bucket": {
                    "type": "score",
                    "instructions": "Total amount",
                    "criteria": ["under 100", "100 to 1,000", "1,000 to 10,000", "over 10,000"],
                },
                "lines_sum_matches_total": {
                    "type": "noul",
                    "instructions": "Does qty x unit_price summed over lines equal total?",
                },
            },
        },
        "structured extraction over a JSON invoice; last question needs arithmetic (expected weak)",
        kind="workload",
    )
    for k, passage in RAG_PASSAGES.items():
        add(
            f"rag_relevance_{k}",
            {
                "state": {"query": RAG_QUERY, "passage": passage},
                "questions": {
                    "relevance": {
                        "type": "score",
                        "instructions": "How relevant is the passage to the query?",
                        "criteria": [
                            "irrelevant",
                            "related topic",
                            "partially answers",
                            "fully answers",
                        ],
                    },
                    "answers": {
                        "type": "noul",
                        "instructions": "Does the passage contain the answer?",
                    },
                },
            },
            "RAG passage scoring (BRIGHT shape); the three passages span fully/related/homonym",
            kind="workload",
        )
    add(
        "moderation",
        {
            "state": COMMENT,
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Which policy category applies, if any?",
                    "criteria": {
                        "harassment": "Targets a person or group with abuse",
                        "threat": "Threatens violence",
                        "spam": "Commercial or repetitive junk",
                        "none": "No policy applies; rude is allowed",
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": "Action needed",
                    "criteria": ["allow", "flag for review", "remove"],
                },
            },
        },
        "moderation with an explicit none option; tests whether 'rude but allowed' lands on none",
        kind="workload",
    )
    base = "Order #A81-2231: card declined, customer needs it before Thursday."
    for suffix, text in (
        ("a", base),
        ("b", base.replace("Thursday", "Tuesday")),
        ("c", base.replace("declined", "approved")),
    ):
        add(
            f"near_dup_{suffix}",
            {
                "state": text,
                "questions": {
                    "payment_ok": {"type": "noul", "instructions": "Did the payment succeed?"},
                    "day": {
                        "type": "choice",
                        "instructions": "Which weekday is the deadline?",
                        "criteria": {"tuesday": None, "thursday": None, "unknown": None},
                    },
                },
            },
            "near-duplicate states differing by one word; catches any cache keyed too coarsely",
            kind="workload",
        )
    return recs


_NOUL_QUESTIONS = [
    "Does the customer mention an order id?",
    "Is the issue about payment?",
    "Is the issue about shipping?",
    "Does the customer state a deadline?",
    "Is the customer angry?",
    "Has the customer contacted their bank?",
    "Does the message mention a gift?",
    "Did the customer try more than once?",
]


def _catalogue(n: int) -> list[str]:
    adjectives = ["brass", "steel", "nylon", "rubber", "copper", "aluminium", "zinc", "ceramic"]
    nouns = [
        "coupling",
        "elbow",
        "gasket",
        "washer",
        "nipple",
        "bushing",
        "adapter",
        "clamp",
        "valve seat",
        "o-ring",
        "union",
        "reducer",
        "tee",
        "cap",
        "plug",
        "hose barb",
    ]
    sizes = ["1/4 in", "3/8 in", "1/2 in", "3/4 in", "1 in"]
    out = []
    i = 0
    while len(out) < n:
        out.append(f"{adjectives[i % 8]} {nouns[(i // 8) % 16]} {sizes[(i // 128) % 5]}")
        i += 1
    return out
