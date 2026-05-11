from engine.constants import SKILL_ALIASES, SORTED_ALIAS_KEYS


def normalize_skills(raw_skills: str) -> list:
    """
    Takes a raw comma-separated skill string, lowercases each token,
    maps it through SKILL_ALIASES (longest match first),
    discards unknowns, and returns a deduplicated list.
    """
    tokens = [t.strip().lower() for t in raw_skills.split(",")]

    canonical = []
    for token in tokens:
        for key in SORTED_ALIAS_KEYS:
            if token == key:
                canonical.append(SKILL_ALIASES[key])
                break

    # Deduplicate preserving insertion order
    seen, deduped = set(), []
    for skill in canonical:
        if skill not in seen:
            seen.add(skill)
            deduped.append(skill)

    return deduped


def build_vocabulary(resume_datasets: list) -> list:
    """
    Creates a sorted alphabetical vocabulary from all
    normalized + deduplicated resume skills.
    """
    vocab = set()
    for resume in resume_datasets:
        vocab.update(normalize_skills(resume["raw_skills"]))
    return sorted(vocab)