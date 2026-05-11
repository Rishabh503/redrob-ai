import math

from engine.normalizer import normalize_skills


def compute_tfidf(resume: dict, vocabulary: list, resume_datasets: list) -> dict:
    """
    TF  = 1 / N         (N = total unique skills after deduplication)
    IDF = ln(10 / df)   (df = number of resumes containing the skill)
    No smoothing — as per problem spec.
    """
    skills = normalize_skills(resume["raw_skills"])
    n = len(skills)

    tfidf = {}
    for skill in vocabulary:
        if skill in skills:
            tf  = 1 / n
            df  = sum(
                1 for r in resume_datasets
                if skill in normalize_skills(r["raw_skills"])
            )
            idf = math.log(10 / df)
            tfidf[skill] = tf * idf
        else:
            tfidf[skill] = 0.0

    return tfidf


def build_jd_vector(job_description: dict, vocabulary: list) -> dict:
    """
    Binary vector over the vocabulary — 1 if skill is required
    or preferred in the JD, 0 otherwise.
    """
    present = (
        set(job_description["required_skills"])
        | set(job_description["preferred_skills"])
    )
    return {skill: (1 if skill in present else 0) for skill in vocabulary}


def cosine_similarity(tfidf: dict, jd_vector: dict, vocabulary: list) -> float:
    """
    Cosine(A, B) = (A · B) / (|A| × |B|)
    A = resume TF-IDF vector, B = JD binary vector.
    """
    dot   = sum(tfidf[s] * jd_vector[s] for s in vocabulary)
    mag_a = math.sqrt(sum(tfidf[s] ** 2 for s in vocabulary))
    mag_b = math.sqrt(sum(jd_vector[s] ** 2 for s in vocabulary))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0