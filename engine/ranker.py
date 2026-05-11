from engine.normalizer import build_vocabulary
from engine.scorer import compute_tfidf, build_jd_vector, cosine_similarity


class ResumeMatchingEngine:
    def __init__(self, resume_datasets: list, job_descriptions: list):
        self.resume_datasets  = resume_datasets
        self.job_descriptions = job_descriptions
        self.vocabulary       = build_vocabulary(resume_datasets)

    def rank(self, job_description: dict) -> list:
        """Returns Top 3 (name, score) tuples for a given JD."""
        jd_vec = build_jd_vector(job_description, self.vocabulary)
        scores = []
        for resume in self.resume_datasets:
            tfidf = compute_tfidf(resume, self.vocabulary, self.resume_datasets)
            score = cosine_similarity(tfidf, jd_vec, self.vocabulary)
            scores.append((resume["name"], score))

        # Descending score; alphabetical name on ties
        scores.sort(key=lambda x: (-round(x[1], 2), x[0]))
        return scores[:3]

    def run(self):
        """Ranks all JDs and prints results in the required output format."""
        for jd in self.job_descriptions:
            label  = jd.get("id", "JD")
            top3   = self.rank(jd)
            result = ", ".join(f"{name}({score:.2f})" for name, score in top3)
            print(f"{label} — {jd['company']} ({jd['role']})")
            print(result)
            print()