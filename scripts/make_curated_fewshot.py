"""
make_curated_fewshot.py
=======================
One-off helper: extract a hand-curated subset of the few-shot example store
into knowledge_base/fewshot/multiclass_examples_curated.json.

The ids below were manually reviewed: each post's content unambiguously
matches its label (the full store contains a noticeable amount of label
noise). 20 posts per label, chosen for variety in length, style
(Reddit posts vs short tweets) and topic.

Ids are kept identical to the source store so the curated subset stays
aligned with the existing few-shot Chroma collection.

Run:
    python scripts/make_curated_fewshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "knowledge_base" / "fewshot" / "multiclass_examples.json"
TARGET = PROJECT_ROOT / "knowledge_base" / "fewshot" / "multiclass_examples_curated.json"

CURATED_IDS: dict[str, list[str]] = {
    # Explicit suicidal ideation, plans, attempts or method-seeking.
    "suicidal": [
        "mc_000000",  # noose tied, dismissed by services
        "mc_000008",  # chronic ideation, driving into a wall
        "mc_000009",  # recent paracetamol attempt
        "mc_000014",  # writing goodbye letters/videos
        "mc_000016",  # "$1,000 before you kill yourself"
        "mc_000017",  # teen: suicide seems the only option
        "mc_000021",  # daily planning, researching methods
        "mc_000024",  # acceptance of suicide as the end
        "mc_000036",  # setting a date
        "mc_000042",  # imminent hanging, posts address
        "mc_000055",  # "killing myself very soon", short
        "mc_000062",  # asking least painful method
        "mc_000074",  # helium tank, tonight
        "mc_000081",  # gap year taken to plan attempt
        "mc_000089",  # "blow my brains out in 30 mins"
        "mc_000117",  # will hang myself soon
        "mc_000120",  # married mother, religious framing, wants out
        "mc_000125",  # 14yo, "getting really close to ending it"
        "mc_000187",  # short prep checklist: finances, suicide letters
        "mc_000196",  # about to jump from a moving car
    ],
    # Clear depressive symptoms (low mood, anhedonia, fatigue, hopelessness)
    # without active suicidal intent.
    "depression": [
        "mc_000200",  # dismissed as a teen phase, starting antidepressants
        "mc_000203",  # small victories (hygiene), recovery framing
        "mc_000220",  # anhedonia despite lifestyle changes
        "mc_000223",  # motivation catch-22
        "mc_000224",  # wasted 20s being depressed
        "mc_000226",  # short: sadness deep in the bones
        "mc_000230",  # cognitive/memory decline from depression
        "mc_000234",  # trapped, worthless, drowning in distractions
        "mc_000235",  # short self-hatred vent
        "mc_000237",  # "does it ever let up?" recovery question
        "mc_000240",  # waking up with morning sadness
        "mc_000244",  # remote-work isolation and loneliness
        "mc_000256",  # masking low mood at work with children
        "mc_000266",  # numbness so bad it is scary
        "mc_000281",  # hypersomnia, slept 20 hours
        "mc_000289",  # jokes outside, miserable inside
        "mc_000298",  # has everything, still empty
        "mc_000303",  # depressed on a Hawaii vacation
        "mc_000307",  # alcohol as coping, college student
        "mc_000352",  # googling "depression signs", inertia
    ],
    # Everyday chatter with no mental-health distress signal.
    "normal": [
        "mc_000405",  # complimenting a new song
        "mc_000410",  # goodnight, replies tomorrow
        "mc_000412",  # invite to play Minecraft/Roblox
        "mc_000424",  # excited for payday and holiday
        "mc_000428",  # asking IB students for advice
        "mc_000433",  # daily screen-time question
        "mc_000434",  # asked out best friend, she said yes
        "mc_000442",  # back after 8 months off reddit
        "mc_000443",  # microwave-at-0:00 joke
        "mc_000450",  # "kill streak" gaming joke (violent words, benign)
        "mc_000460",  # hoping a covid swab is negative
        "mc_000468",  # dinosaur facts
        "mc_000496",  # taking socks off before sleep
        "mc_000503",  # Christmas gift ideas for parents
        "mc_000504",  # new year's plans
        "mc_000508",  # complaint about teachers / sleeping in class
        "mc_000513",  # busy working on a film
        "mc_000514",  # asking how to get rid of mice
        "mc_000540",  # mild reluctance about school tomorrow
        "mc_000596",  # working on a lab report
    ],
}


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in rows}

    curated: list[dict] = []
    for label, ids in CURATED_IDS.items():
        assert len(ids) == 20, f"{label}: expected 20 ids, got {len(ids)}"
        for row_id in ids:
            row = by_id[row_id]
            assert row["label"] == label, (
                f"{row_id}: source label {row['label']!r} != expected {label!r}"
            )
            curated.append(row)

    TARGET.write_text(
        json.dumps(curated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {TARGET} ({len(curated)} examples)")


if __name__ == "__main__":
    main()
