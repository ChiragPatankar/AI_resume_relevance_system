"""
Similarity calculation module for resume-job description matching.

This module provides the SimilarityCalculator class for computing various
similarity metrics between resumes and job descriptions using TF-IDF,
semantic embeddings, and keyword matching approaches.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import re

# ML and similarity libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

from config import config
from .text_processor import TextProcessor


class SimilarityCalculator:
    """
    Advanced similarity calculator for resume-job description matching.
    
    Implements multiple similarity metrics including TF-IDF, semantic embeddings,
    keyword matching, and skill-specific similarities optimized for recruitment.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        self.logger = logging.getLogger(__name__)
        self.config = config.model
        self.text_processor = TextProcessor()
        
        # Initialize models
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize TF-IDF and sentence transformer models."""
        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.config.tfidf_max_features,
                    ngram_range=self.config.tfidf_ngram_range,
                    stop_words=self.config.tfidf_stop_words,
                    lowercase=True,
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9\+\#\.\-]*[a-zA-Z0-9]\b|[a-zA-Z]'
                )
                self.logger.info("TF-IDF vectorizer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize TF-IDF vectorizer: {str(e)}")
        
        # Initialize sentence transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer(
                    self.config.sentence_transformer_model
                )
                self.logger.info(f"Sentence transformer loaded: {self.config.sentence_transformer_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {str(e)}")
                self.sentence_transformer = None
    
    def calculate_comprehensive_similarity(
        self, 
        resume_text: str, 
        job_description_text: str,
        resume_skills: List[str] = None,
        job_skills: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive similarity using multiple methods.
        
        Args:
            resume_text: Resume text content
            job_description_text: Job description text content
            resume_skills: List of resume skills
            job_skills: List of job required skills
            
        Returns:
            Dictionary with various similarity scores
        """
        results = {
            'tfidf_similarity': 0.0,
            'semantic_similarity': 0.0,
            'keyword_similarity': 0.0,
            'skill_similarity': 0.0,
            'overall_similarity': 0.0,
            'confidence': 0.0
        }
        
        try:
            # Preprocess texts
            resume_processed = self.text_processor.preprocess_text(resume_text)
            job_processed = self.text_processor.preprocess_text(job_description_text)
            
            if not resume_processed or not job_processed:
                self.logger.warning("Empty text after preprocessing")
                return results
            
            # Calculate TF-IDF similarity
            tfidf_score = self._calculate_tfidf_similarity(resume_processed, job_processed)
            results['tfidf_similarity'] = tfidf_score
            
            # Calculate semantic similarity
            semantic_score = self._calculate_semantic_similarity(resume_text, job_description_text)
            results['semantic_similarity'] = semantic_score
            
            # Calculate keyword similarity
            keyword_score = self._calculate_keyword_similarity(resume_processed, job_processed)
            results['keyword_similarity'] = keyword_score
            
            # Calculate skill similarity
            if resume_skills and job_skills:
                skill_score = self._calculate_skill_similarity(resume_skills, job_skills)
                results['skill_similarity'] = skill_score
            
            # Calculate overall similarity using weighted combination
            weights = config.scoring
            overall_score = (
                tfidf_score * 0.3 +
                semantic_score * 0.4 +
                keyword_score * 0.2 +
                results['skill_similarity'] * 0.1
            )
            results['overall_similarity'] = overall_score
            
            # Calculate confidence based on available methods
            confidence = self._calculate_confidence(results)
            results['confidence'] = confidence
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive similarity calculation: {str(e)}")
        
        return results
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity between two texts."""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            return 0.0
        
        try:
            # Fit and transform both texts
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity_score = similarity_matrix[0][0]
            
            return float(similarity_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
            return 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not self.sentence_transformer:
            # Fallback to spaCy similarity if available
            return self.text_processor.calculate_text_similarity(text1, text2)
        
        try:
            # Limit text length for performance
            text1_truncated = text1[:2000]
            text2_truncated = text2[:2000]
            
            # Generate embeddings
            embeddings = self.sentence_transformer.encode([text1_truncated, text2_truncated])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword-based similarity."""
        try:
            # Extract keywords from both texts
            keywords1 = self.text_processor.extract_keywords(text1, top_k=50)
            keywords2 = self.text_processor.extract_keywords(text2, top_k=50)
            
            # Get keyword sets
            kw_set1 = set(kw['keyword'] for kw in keywords1)
            kw_set2 = set(kw['keyword'] for kw in keywords2)
            
            if not kw_set1 or not kw_set2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = kw_set1.intersection(kw_set2)
            union = kw_set1.union(kw_set2)
            
            jaccard_score = len(intersection) / len(union) if union else 0.0
            
            # Weight by keyword importance (TF scores)
            weighted_score = 0.0
            total_weight = 0.0
            
            kw_dict1 = {kw['keyword']: kw['tf_score'] for kw in keywords1}
            kw_dict2 = {kw['keyword']: kw['tf_score'] for kw in keywords2}
            
            for keyword in intersection:
                weight = min(kw_dict1.get(keyword, 0), kw_dict2.get(keyword, 0))
                weighted_score += weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_similarity = weighted_score / total_weight
                # Combine Jaccard and weighted similarity
                return (jaccard_score + weighted_similarity) / 2
            else:
                return jaccard_score
                
        except Exception as e:
            self.logger.error(f"Error calculating keyword similarity: {str(e)}")
            return 0.0
    
    def _calculate_skill_similarity(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill-specific similarity."""
        if not resume_skills or not job_skills:
            return 0.0
        
        try:
            resume_skills_lower = [skill.lower() for skill in resume_skills]
            job_skills_lower = [skill.lower() for skill in job_skills]
            
            # Exact matches
            exact_matches = set(resume_skills_lower).intersection(set(job_skills_lower))
            exact_score = len(exact_matches) / len(job_skills_lower)
            
            # Fuzzy matches for remaining skills
            fuzzy_score = 0.0
            if FUZZYWUZZY_AVAILABLE:
                unmatched_job_skills = [skill for skill in job_skills_lower if skill not in exact_matches]
                unmatched_resume_skills = [skill for skill in resume_skills_lower if skill not in exact_matches]
                
                fuzzy_matches = 0
                for job_skill in unmatched_job_skills:
                    best_match_score = max(
                        [fuzz.ratio(job_skill, resume_skill) for resume_skill in unmatched_resume_skills],
                        default=0
                    )
                    if best_match_score >= 80:  # 80% similarity threshold
                        fuzzy_matches += (best_match_score / 100.0)
                
                if unmatched_job_skills:
                    fuzzy_score = fuzzy_matches / len(unmatched_job_skills)
            
            # Combine exact and fuzzy scores
            total_score = (exact_score * 0.8) + (fuzzy_score * 0.2)
            
            return min(total_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating skill similarity: {str(e)}")
            return 0.0
    
    def _calculate_confidence(self, results: Dict[str, float]) -> float:
        """Calculate confidence score based on available similarity metrics."""
        available_scores = []
        
        if results['tfidf_similarity'] > 0:
            available_scores.append(results['tfidf_similarity'])
        
        if results['semantic_similarity'] > 0:
            available_scores.append(results['semantic_similarity'])
        
        if results['keyword_similarity'] > 0:
            available_scores.append(results['keyword_similarity'])
        
        if results['skill_similarity'] > 0:
            available_scores.append(results['skill_similarity'])
        
        if not available_scores:
            return 0.0
        
        # Confidence based on:
        # 1. Number of available metrics
        # 2. Consistency of scores (low variance = high confidence)
        # 3. Overall score magnitude
        
        num_metrics = len(available_scores)
        max_metrics = 4
        
        # Base confidence from number of metrics
        metric_confidence = num_metrics / max_metrics
        
        # Consistency confidence (inverse of variance)
        if len(available_scores) > 1:
            variance = np.var(available_scores)
            consistency_confidence = 1.0 / (1.0 + variance * 10)  # Scale variance
        else:
            consistency_confidence = 0.8  # Single metric has medium confidence
        
        # Magnitude confidence (higher scores generally more reliable)
        magnitude_confidence = min(results['overall_similarity'] * 1.5, 1.0)
        
        # Weighted combination
        confidence = (
            metric_confidence * 0.4 +
            consistency_confidence * 0.4 +
            magnitude_confidence * 0.2
        )
        
        return min(confidence, 1.0)
    
    def calculate_section_similarities(
        self,
        resume_sections: Dict[str, str],
        job_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate similarities for specific resume/job sections.
        
        Args:
            resume_sections: Dictionary of resume sections
            job_sections: Dictionary of job sections
            
        Returns:
            Dictionary of section-wise similarities
        """
        similarities = {}
        
        # Common section mappings
        section_mappings = {
            'skills': ['skills', 'technical_skills', 'core_competencies'],
            'experience': ['experience', 'work_experience', 'employment'],
            'education': ['education', 'academic_background'],
            'projects': ['projects', 'key_projects'],
            'responsibilities': ['responsibilities', 'duties', 'role']
        }
        
        for section_type, section_names in section_mappings.items():
            resume_text = ""
            job_text = ""
            
            # Aggregate text from matching sections
            for name in section_names:
                if name in resume_sections:
                    resume_text += " " + resume_sections[name]
                if name in job_sections:
                    job_text += " " + job_sections[name]
            
            if resume_text.strip() and job_text.strip():
                similarity = self._calculate_tfidf_similarity(
                    resume_text.strip(), 
                    job_text.strip()
                )
                similarities[section_type] = similarity
            else:
                similarities[section_type] = 0.0
        
        return similarities
    
    def get_similarity_explanation(self, results: Dict[str, float]) -> Dict[str, str]:
        """
        Generate human-readable explanations for similarity scores.
        
        Args:
            results: Similarity calculation results
            
        Returns:
            Dictionary with explanations for each score
        """
        explanations = {}
        
        def score_to_description(score: float) -> str:
            if score >= 0.9:
                return "Excellent match"
            elif score >= 0.8:
                return "Very good match"
            elif score >= 0.7:
                return "Good match"
            elif score >= 0.6:
                return "Fair match"
            elif score >= 0.5:
                return "Moderate match"
            elif score >= 0.3:
                return "Poor match"
            else:
                return "Very poor match"
        
        explanations['tfidf'] = (
            f"TF-IDF similarity: {results['tfidf_similarity']:.3f} "
            f"({score_to_description(results['tfidf_similarity'])}). "
            f"Measures overlap of important terms and phrases."
        )
        
        explanations['semantic'] = (
            f"Semantic similarity: {results['semantic_similarity']:.3f} "
            f"({score_to_description(results['semantic_similarity'])}). "
            f"Measures meaning and context similarity using AI."
        )
        
        explanations['keyword'] = (
            f"Keyword similarity: {results['keyword_similarity']:.3f} "
            f"({score_to_description(results['keyword_similarity'])}). "
            f"Measures overlap of key terms and concepts."
        )
        
        explanations['skill'] = (
            f"Skill similarity: {results['skill_similarity']:.3f} "
            f"({score_to_description(results['skill_similarity'])}). "
            f"Measures direct skill matching between resume and job requirements."
        )
        
        explanations['overall'] = (
            f"Overall similarity: {results['overall_similarity']:.3f} "
            f"({score_to_description(results['overall_similarity'])}). "
            f"Weighted combination of all similarity measures."
        )
        
        explanations['confidence'] = (
            f"Confidence: {results['confidence']:.3f}. "
            f"Reliability of the similarity assessment based on data quality and consistency."
        )
        
        return explanations
    
    def find_matching_keywords(self, text1: str, text2: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find keywords that appear in both texts.
        
        Args:
            text1: First text
            text2: Second text
            top_k: Number of top matches to return
            
        Returns:
            List of matching keywords with scores
        """
        try:
            keywords1 = self.text_processor.extract_keywords(text1, top_k=100)
            keywords2 = self.text_processor.extract_keywords(text2, top_k=100)
            
            kw_dict1 = {kw['keyword']: kw for kw in keywords1}
            kw_dict2 = {kw['keyword']: kw for kw in keywords2}
            
            matches = []
            
            for keyword in kw_dict1:
                if keyword in kw_dict2:
                    matches.append({
                        'keyword': keyword,
                        'resume_score': kw_dict1[keyword]['tf_score'],
                        'job_score': kw_dict2[keyword]['tf_score'],
                        'combined_score': (kw_dict1[keyword]['tf_score'] + kw_dict2[keyword]['tf_score']) / 2,
                        'is_skill': kw_dict1[keyword]['is_skill']
                    })
            
            # Sort by combined score
            matches.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return matches[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding matching keywords: {str(e)}")
            return []
    
    def calculate_skill_gap_analysis(
        self, 
        resume_skills: List[str], 
        required_skills: List[str],
        preferred_skills: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze skill gaps between resume and job requirements.
        
        Args:
            resume_skills: Skills found in resume
            required_skills: Required skills for job
            preferred_skills: Preferred skills for job
            
        Returns:
            Detailed skill gap analysis
        """
        if preferred_skills is None:
            preferred_skills = []
        
        analysis = {
            'matched_required': [],
            'missing_required': [],
            'matched_preferred': [],
            'missing_preferred': [],
            'additional_skills': [],
            'required_skill_coverage': 0.0,
            'preferred_skill_coverage': 0.0,
            'total_skill_score': 0.0
        }
        
        try:
            resume_skills_lower = set(skill.lower() for skill in resume_skills)
            required_skills_lower = [skill.lower() for skill in required_skills]
            preferred_skills_lower = [skill.lower() for skill in preferred_skills]
            
            # Analyze required skills
            for skill in required_skills_lower:
                skill_matched = False
                
                # Direct match
                if skill in resume_skills_lower:
                    analysis['matched_required'].append(skill)
                    skill_matched = True
                else:
                    # Check for fuzzy matches
                    if FUZZYWUZZY_AVAILABLE:
                        best_match_score = 0
                        best_match_skill = None
                        
                        for resume_skill in resume_skills_lower:
                            match_score = fuzz.ratio(skill, resume_skill)
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_match_skill = resume_skill
                        
                        if best_match_score >= 80:
                            analysis['matched_required'].append(f"{skill} (similar to {best_match_skill})")
                            skill_matched = True
                
                # Only add to missing if not matched
                if not skill_matched:
                    analysis['missing_required'].append(skill)
            
            # Analyze preferred skills
            for skill in preferred_skills_lower:
                if skill in resume_skills_lower:
                    analysis['matched_preferred'].append(skill)
                else:
                    analysis['missing_preferred'].append(skill)
            
            # Find additional skills
            all_job_skills = set(required_skills_lower + preferred_skills_lower)
            for skill in resume_skills_lower:
                if skill not in all_job_skills:
                    analysis['additional_skills'].append(skill)
            
            # Calculate coverage percentages
            if required_skills_lower:
                analysis['required_skill_coverage'] = len(analysis['matched_required']) / len(required_skills_lower)
            
            if preferred_skills_lower:
                analysis['preferred_skill_coverage'] = len(analysis['matched_preferred']) / len(preferred_skills_lower)
            
            # Calculate total skill score
            required_weight = 0.8
            preferred_weight = 0.2
            
            analysis['total_skill_score'] = (
                analysis['required_skill_coverage'] * required_weight +
                analysis['preferred_skill_coverage'] * preferred_weight
            )
            
        except Exception as e:
            self.logger.error(f"Error in skill gap analysis: {str(e)}")
        
        return analysis

