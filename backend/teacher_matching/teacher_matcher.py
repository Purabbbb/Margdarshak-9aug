"""
teacher_matcher.py
------------------
Hybrid Teacher / Expert Matching Engine.

Implements the multi-criteria ranking algorithm:
  0.40 * Skill Gap Coverage
+ 0.30 * Semantic Similarity (Fast Scikit-Learn TF-IDF N-gram Matching)
+ 0.15 * Role / Domain Relevance
+ 0.10 * Experience Score
+ 0.05 * Rating Score

Features:
- Canonical skill normalization reusing the skills database.
- Explainable recommendation breakdowns.
- Pure scikit-learn TF-IDF semantic matching (zero PyTorch/Transformer bloat).
- Pre-filtering to discard completely irrelevant candidates.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from backend.teacher_matching.schemas import (
        TeacherProfile,
        TeacherPreferences,
        TeacherRecommendation,
        TeacherRecommendResponse
    )
    from backend.teacher_matching.teacher_repository import TeacherRepository
    from backend.data.skills_db import SKILLS_DB
    from backend.data.job_roles import JOB_ROLES
except ModuleNotFoundError:
    from teacher_matching.schemas import (
        TeacherProfile,
        TeacherPreferences,
        TeacherRecommendation,
        TeacherRecommendResponse
    )
    from teacher_matching.teacher_repository import TeacherRepository
    from data.skills_db import SKILLS_DB
    from data.job_roles import JOB_ROLES

# ----------------- Skill Normalization Map -----------------
# Flatten canonical skills from SKILLS_DB
ALL_CANONICAL_SKILLS = set()
for category_skills in SKILLS_DB.values():
    for s in category_skills:
        ALL_CANONICAL_SKILLS.add(s.lower())

COMMON_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ai": "machine learning",
    "genai": "generative ai",
    "powerbi": "power bi",
    "pbi": "power bi",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "expressjs": "express",
    "vuejs": "vue",
    "nextjs": "next.js",
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp cloud": "gcp",
    "google cloud platform": "gcp",
    "bi": "business intelligence",
    "stats": "statistics",
    "pen testing": "penetration testing",
    "pentesting": "penetration testing",
    "ethical hack": "ethical hacking",
    "cicd": "ci/cd",
    "ci / cd": "ci/cd",
    "ui/ux": "ui design",
    "ux": "ux design",
    "ui": "ui design",
}

def normalize_skill(skill: str) -> str:
    """Normalize a raw skill string to the canonical terminology."""
    cleaned = skill.strip().lower()
    if cleaned in COMMON_ALIASES:
        return COMMON_ALIASES[cleaned]
    # Remove extra symbols
    cleaned_simplified = re.sub(r"[^\w\s\.\+#/-]", "", cleaned)
    if cleaned_simplified in COMMON_ALIASES:
        return COMMON_ALIASES[cleaned_simplified]
    if cleaned in ALL_CANONICAL_SKILLS:
        return cleaned
    return cleaned


# Role to domain associations
ROLE_DOMAINS_MAP = {
    "Data Scientist": ["data science", "machine learning", "statistics"],
    "Data Analyst": ["data analytics", "business intelligence", "databases"],
    "Machine Learning Engineer": ["machine learning", "deep learning", "cloud_devops"],
    "Frontend Developer": ["web_frontend", "ui_ux"],
    "Backend Developer": ["web_backend", "databases"],
    "Full Stack Developer": ["web_frontend", "web_backend", "databases"],
    "DevOps Engineer": ["cloud_devops"],
    "Cybersecurity Analyst": ["cybersecurity"],
    "Mobile App Developer": ["mobile", "web_frontend"],
    "UI/UX Designer": ["ui_ux", "web_frontend"],
    "Business Analyst": ["business_analysis", "project_management", "data analytics"],
    "Cloud Architect": ["cloud_devops", "databases"]
}


class TeacherMatcher:
    def __init__(self, repository: Optional[TeacherRepository] = None):
        self.repo = repository or TeacherRepository()

    def _calculate_semantic_scores(
        self,
        query: str,
        teachers: List[TeacherProfile]
    ) -> Dict[str, float]:
        """
        Calculates semantic similarity between the student query and each teacher profile.
        Uses TF-IDF n-gram vectorization with cosine similarity for fast, deterministic,
        reliable execution across all platforms.
        """
        teacher_docs = [
            f"{t.headline}. Domains: {' '.join(t.domains)}. Skills: {' '.join(t.skills)}. Bio: {t.bio}. Style: {' '.join(t.teaching_style)}."
            for t in teachers
        ]

        try:
            corpus = [query] + teacher_docs
            tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(corpus)
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            return {
                teachers[i].id: float(np.clip(cos_sim[i] * 100.0, 0.0, 100.0))
                for i in range(len(teachers))
            }
        except Exception:
            # Fallback uniform base
            return {t.id: 50.0 for t in teachers}

    def _calculate_role_relevance(self, teacher: TeacherProfile, target_role: str) -> float:
        """Score how closely a teacher's domains and skill set match the target role."""
        expected_domains = [d.lower() for d in ROLE_DOMAINS_MAP.get(target_role, [])]
        teacher_domains = [d.lower() for d in teacher.domains]

        # Domain overlap
        domain_overlap = set(expected_domains) & set(teacher_domains)
        if domain_overlap:
            domain_score = min(100.0, 70.0 + (len(domain_overlap) * 15.0))
        else:
            domain_score = 40.0

        # Target role skill alignment
        role_info = JOB_ROLES.get(target_role, {})
        required_role_skills = [s.lower() for s in role_info.get("required", [])]
        overlap_with_role = set(teacher.skills) & set(required_role_skills)
        if required_role_skills:
            skill_role_score = (len(overlap_with_role) / len(required_role_skills)) * 100.0
        else:
            skill_role_score = 50.0

        return round(0.5 * domain_score + 0.5 * skill_role_score, 1)

    def _generate_explanation(
        self,
        teacher: TeacherProfile,
        target_role: str,
        matched_skills: List[str],
        total_gaps_count: int,
        skill_coverage: float,
        role_relevance: float
    ) -> str:
        """Create a clear, transparent, human-readable justification for the recommendation."""
        matched_sample = ", ".join(s.title() for s in matched_skills[:3])

        if skill_coverage >= 75 and matched_sample:
            return (
                f"Top mentor for your gaps — specializes in {matched_sample}, "
                f"covering {len(matched_skills)} of your {total_gaps_count} missing skills "
                f"for the {target_role} track with {teacher.experience_years} years industry experience."
            )
        elif matched_skills:
            return (
                f"Directly teaches {matched_sample} with hands-on {teacher.teaching_style[0] if teacher.teaching_style else 'mentorship'}, "
                f"bringing {teacher.teaching_experience_years} years teaching experience in {teacher.domains[0].replace('_', ' ').title()}."
            )
        else:
            return (
                f"Strong foundational mentor for the {target_role} domain ({', '.join(teacher.domains).replace('_', ' ').title()}) "
                f"rated {teacher.rating}/5.0 across {teacher.sessions_completed} student sessions."
            )

    def recommend_teachers(
        self,
        analysis: Dict[str, Any],
        target_role: Optional[str] = None,
        preferences: Optional[TeacherPreferences] = None,
        limit: int = 6
    ) -> TeacherRecommendResponse:
        """
        Executes the hybrid ranking pipeline for a student's resume analysis.
        """
        # 1. Determine target role
        job_roles = analysis.get("job_roles", {})
        if not target_role:
            target_role = job_roles.get("best_match", {}).get("role", "Software Engineer")

        # 2. Extract and normalize skill gaps
        gaps_dict = analysis.get("gaps", {}).get("gaps", {})
        role_gaps = gaps_dict.get(target_role, {})
        missing_required = [normalize_skill(s) for s in role_gaps.get("missing_required", [])]
        missing_preferred = [normalize_skill(s) for s in role_gaps.get("missing_preferred", [])]

        # Combine gaps with priority to required
        raw_gaps = missing_required + [s for s in missing_preferred if s not in missing_required]
        # De-duplicate while preserving order
        student_gaps = []
        for g in raw_gaps:
            if g and g not in student_gaps:
                student_gaps.append(g)

        # Extracted skills from resume
        matched_resume_skills = analysis.get("skills", {}).get("matched_skills", [])
        current_skills = [normalize_skill(s) for s in matched_resume_skills]

        # 3. Filter candidates by user preferences
        mode_pref = preferences.mode if preferences else None
        lang_pref = preferences.language if preferences else None
        style_pref = preferences.teaching_style if preferences else None

        candidates = self.repo.filter_teachers(
            mode=mode_pref,
            language=lang_pref,
            teaching_style=style_pref
        )

        # Fallback if preferences eliminated everyone
        if not candidates:
            candidates = self.repo.get_all()

        # 4. Pre-filtering: Prioritize teachers with domain or skill overlap
        gap_set = set(student_gaps)
        role_domains = set(ROLE_DOMAINS_MAP.get(target_role, []))

        relevant_pool = []
        for t in candidates:
            teacher_skills = {
                normalize_skill(skill)
                for skill in (t.top_skills or t.skills[:5])[:5]
            }
            teacher_domains = set(t.domains)
            has_gap_overlap = bool(teacher_skills & gap_set)
            has_domain_overlap = bool(teacher_domains & role_domains)
            if has_gap_overlap or has_domain_overlap:
                relevant_pool.append(t)

        # Fallback to all candidates if pool is too small
        if len(relevant_pool) < 3:
            relevant_pool = candidates

        # 5. Build student query for semantic comparison
        query = (
            f"Student seeking guidance to become a {target_role}. "
            f"Missing skills: {', '.join(student_gaps) if student_gaps else 'none'}. "
            f"Current skills: {', '.join(current_skills[:10]) if current_skills else 'programming'}."
        )

        semantic_scores = self._calculate_semantic_scores(query, relevant_pool)

        # 6. Compute scores for each candidate
        recommendations = []
        for teacher in relevant_pool:
            t_skills = {
                normalize_skill(skill)
                for skill in (teacher.top_skills or teacher.skills[:5])[:5]
            }

            # A. Skill Gap Coverage (0 - 100)
            if student_gaps:
                matched = [g for g in student_gaps if g in t_skills]
                missing = [g for g in student_gaps if g not in t_skills]
                skill_coverage = (len(matched) / len(student_gaps)) * 100.0
            else:
                # If student has no gaps for this role, score based on preferred/role skills
                role_skills = JOB_ROLES.get(target_role, {}).get("required", [])
                matched = [s for s in role_skills if s in t_skills]
                missing = [s for s in role_skills if s not in t_skills]
                skill_coverage = (len(matched) / max(len(role_skills), 1)) * 100.0

            # B. Semantic Similarity (0 - 100)
            sem_score = semantic_scores.get(teacher.id, 50.0)

            # C. Role / Domain Relevance (0 - 100)
            role_rel = self._calculate_role_relevance(teacher, target_role)

            # D. Experience Score (0 - 100)
            # 10+ years = 100, 3 years = 60
            exp_score = min(100.0, 50.0 + (teacher.experience_years * 5.0))

            # E. Rating Score (0 - 100)
            rating_score = (teacher.rating / 5.0) * 100.0

            # Weighted Hybrid Score
            overall = (
                0.40 * skill_coverage
                + 0.30 * sem_score
                + 0.15 * role_rel
                + 0.10 * exp_score
                + 0.05 * rating_score
            )

            explanation = self._generate_explanation(
                teacher=teacher,
                target_role=target_role,
                matched_skills=matched,
                total_gaps_count=len(student_gaps) or len(matched),
                skill_coverage=skill_coverage,
                role_relevance=role_rel
            )

            rec = TeacherRecommendation(
                rank=0,  # Will be assigned after sort
                teacher=teacher,
                overall_match=round(overall, 1),
                skill_coverage=round(skill_coverage, 1),
                matched_skills=matched,
                missing_teacher_coverage=missing,
                semantic_similarity=round(sem_score, 1),
                role_relevance=round(role_rel, 1),
                experience_score=round(exp_score, 1),
                rating_score=round(rating_score, 1),
                reason=explanation
            )
            recommendations.append(rec)

        # 7. Sort by overall_match descending, then rating descending
        recommendations.sort(
            key=lambda x: (x.overall_match, x.skill_coverage, x.teacher.rating),
            reverse=True
        )

        # 8. Assign ranks and truncate to limit
        top_recs = recommendations[:limit]
        for i, rec in enumerate(top_recs, start=1):
            rec.rank = i

        method_note = "hybrid semantic (TF-IDF N-gram) + skill coverage (0.40) + role relevance (0.15) ranking"

        return TeacherRecommendResponse(
            status="success",
            target_role=target_role,
            skill_gaps=student_gaps,
            recommendations=top_recs,
            method=method_note
        )
