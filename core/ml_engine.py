"""
SafeNest AI - Core Machine Learning & Recommendation Engine
Authors: AI/ML Engineering Lead
Description:
    1. Custom TF-IDF Vectorizer & Cosine Similarity search over natural language student preferences.
    2. Multi-Criteria Safety Index (weighted feature scoring & normalization).
    3. Hybrid Composite Matcher balancing semantic preference match, safety index, and budget.
"""

import json
import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np


class SimpleTFIDFRecommender:
    """
    Lightweight TF-IDF Vectorizer and Cosine Similarity Matcher implemented
    using NumPy vector linear algebra.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf_vector: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        self.corpus_size: int = 0
        self.items: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and clean alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        # Filter basic stopwords
        stopwords = {
            "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to",
            "was", "were", "will", "with", "i", "want", "need", "pg", "looking"
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def fit(self, items: List[Dict[str, Any]]):
        """Build vocabulary, calculate IDF, and precompute item document vectors."""
        self.items = items
        self.corpus_size = len(items)

        # Build document representation combining name, description, features, and location
        documents = []
        for item in items:
            combined = f"{item['name']} {item['location']} {item['room_type']} {item['gender_type']} {item['description']} {' '.join(item['features'])}"
            documents.append(self._tokenize(combined))

        # Build vocabulary
        vocab_set = set()
        for doc in documents:
            vocab_set.update(doc)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        vocab_len = len(self.vocabulary)

        # Calculate Document Frequency (DF)
        df = np.zeros(vocab_len, dtype=np.float64)
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                if term in self.vocabulary:
                    df[self.vocabulary[term]] += 1.0

        # Calculate Smooth Inverse Document Frequency (IDF) = log((N + 1) / (DF + 1)) + 1
        self.idf_vector = np.log((self.corpus_size + 1.0) / (df + 1.0)) + 1.0

        # Vectorize all corpus documents
        doc_vectors = []
        for doc in documents:
            tf = np.zeros(vocab_len, dtype=np.float64)
            for term in doc:
                if term in self.vocabulary:
                    tf[self.vocabulary[term]] += 1.0
            # Term Frequency normalization (sublinear scaling: 1 + log(tf) if tf > 0)
            tf_scaled = np.where(tf > 0, 1.0 + np.log(tf + 1e-9), 0.0)
            tfidf = tf_scaled * self.idf_vector
            # L2 Normalization (Cosine unit vector)
            norm = np.linalg.norm(tfidf)
            if norm > 0:
                tfidf = tfidf / norm
            doc_vectors.append(tfidf)

        self.doc_vectors = np.array(doc_vectors)

    def query_similarity(self, query_text: str) -> np.ndarray:
        """Compute cosine similarity between query and all stored documents."""
        if self.corpus_size == 0 or len(self.vocabulary) == 0:
            return np.zeros(self.corpus_size)

        tokens = self._tokenize(query_text)
        if not tokens:
            # If no recognizable tokens, return uniform baseline
            return np.ones(self.corpus_size) * 0.5

        vocab_len = len(self.vocabulary)
        q_tf = np.zeros(vocab_len, dtype=np.float64)
        for term in tokens:
            if term in self.vocabulary:
                q_tf[self.vocabulary[term]] += 1.0

        q_scaled = np.where(q_tf > 0, 1.0 + np.log(q_tf + 1e-9), 0.0)
        q_tfidf = q_scaled * self.idf_vector
        q_norm = np.linalg.norm(q_tfidf)
        if q_norm > 0:
            q_tfidf = q_tfidf / q_norm

        # Cosine similarity = dot product of unit vectors
        similarities = np.dot(self.doc_vectors, q_tfidf)
        return np.clip(similarities, 0.0, 1.0)


class SafetyIndexCalculator:
    """
    Explainable Multi-Criteria Safety Model.
    Computes a normalized Safety Index (0 - 100%) with feature contribution breakdowns.
    """

    FEATURE_WEIGHTS = {
        "security_guard_24x7": 0.25,     # 25% weight
        "cctv_coverage": 0.20,           # 20% weight (scaled 0-100 -> 0-1)
        "biometric_entry": 0.15,         # 15% weight
        "resident_warden": 0.15,         # 15% weight
        "street_lighting_rating": 0.10,  # 10% weight (scaled 1-5 -> 0-1)
        "emergency_alarm_installed": 0.10, # 10% weight
        "proximity_factor": 0.05         # 5% weight (<1km gets max score)
    }

    @classmethod
    def calculate_score(cls, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
            total_score (0.0 to 100.0)
            breakdown: dictionary with per-factor percentage contribution
        """
        breakdown = {}

        # 1. 24x7 Security
        guard_val = float(metrics.get("security_guard_24x7", 0))
        breakdown["24/7 Security Guard"] = guard_val * cls.FEATURE_WEIGHTS["security_guard_24x7"] * 100

        # 2. CCTV Coverage (0-100)
        cctv_val = min(1.0, max(0.0, float(metrics.get("cctv_coverage", 0)) / 100.0))
        breakdown["CCTV Surveillance"] = cctv_val * cls.FEATURE_WEIGHTS["cctv_coverage"] * 100

        # 3. Biometric / Smart Entry
        bio_val = float(metrics.get("biometric_entry", 0))
        breakdown["Biometric / Smart Entry"] = bio_val * cls.FEATURE_WEIGHTS["biometric_entry"] * 100

        # 4. Resident Female Warden
        warden_val = float(metrics.get("resident_warden", 0))
        breakdown["Resident Warden"] = warden_val * cls.FEATURE_WEIGHTS["resident_warden"] * 100

        # 5. Street Lighting Rating (1 to 5 stars)
        light_val = min(1.0, max(0.0, float(metrics.get("street_lighting_rating", 3.0)) / 5.0))
        breakdown["Street Lighting & Road Safety"] = light_val * cls.FEATURE_WEIGHTS["street_lighting_rating"] * 100

        # 6. Emergency SOS Alarm
        alarm_val = float(metrics.get("emergency_alarm_installed", 0))
        breakdown["Emergency SOS / Alarms"] = alarm_val * cls.FEATURE_WEIGHTS["emergency_alarm_installed"] * 100

        # 7. Proximity to Campus (Closer = safer night walking)
        dist_km = float(metrics.get("distance_to_campus_km", 2.0))
        # Decay function: <= 0.5km = 1.0, 3km = 0.2
        prox_val = max(0.1, min(1.0, 1.0 - (dist_km - 0.5) / 2.5)) if dist_km > 0.5 else 1.0
        breakdown["Campus Proximity Safety"] = prox_val * cls.FEATURE_WEIGHTS["proximity_factor"] * 100

        total_score = sum(breakdown.values())
        return round(total_score, 1), breakdown


class SafeNestRecommenderEngine:
    """
    Orchestrator combining Semantic Search, Multi-Factor Safety, and Budget Constraints.
    """

    def __init__(self, dataset_path: str):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.nlp_engine = SimpleTFIDFRecommender()
        self.nlp_engine.fit(self.items)

    def recommend(
        self,
        query: str = "",
        max_budget: int = 15000,
        min_safety_score: float = 50.0,
        alpha_semantic: float = 0.5,
        alpha_safety: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid ranking algorithm:
            Composite Score = (alpha_semantic * Semantic_Sim) + (alpha_safety * (Safety_Score / 100))
        """
        # Step 1: Compute NLP similarities
        similarities = self.nlp_engine.query_similarity(query) if query.strip() else np.ones(len(self.items)) * 0.7

        results = []
        for idx, item in enumerate(self.items):
            # Compute safety score
            safety_score, breakdown = SafetyIndexCalculator.calculate_score(item["safety_metrics"])

            # Filter by hard constraints
            if item["monthly_rent"] > max_budget:
                continue
            if safety_score < min_safety_score:
                continue

            sim_score = float(similarities[idx])
            # Composite rank
            composite_score = (alpha_semantic * sim_score) + (alpha_safety * (safety_score / 100.0))
            composite_pct = round(composite_score * 100, 1)

            result_entry = dict(item)
            result_entry["safety_score"] = safety_score
            result_entry["safety_breakdown"] = breakdown
            result_entry["semantic_similarity_pct"] = round(sim_score * 100, 1)
            result_entry["composite_match_score"] = composite_pct
            results.append(result_entry)

        # Sort descending by composite match score
        results.sort(key=lambda x: x["composite_match_score"], reverse=True)
        return results
