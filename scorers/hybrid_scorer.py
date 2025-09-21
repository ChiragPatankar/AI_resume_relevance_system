"""
Hybrid scoring system for resume-job description matching.

This module implements the HybridScorer class which combines rule-based
and ML-based scoring approaches to evaluate resume relevance against
job descriptions for Innomatics Research Labs.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re

from models.resume_data import ResumeData
from models.job_description import JobDescription
from models.scoring_result import ScoringResult, ScoreBreakdown, ScoreExplanation
from processors.text_processor import TextProcessor
from processors.similarity_calculator import SimilarityCalculator
from processors.groq_enhancer import GroqEnhancer
from config import config


class HybridScorer:
    """
    Hybrid scorer combining rule-based and ML-based approaches.
    
    Implements the weighted scoring system for Innomatics placement requirements:
    - Skills matching (40%)
    - Experience relevance (35%) 
    - Education alignment (15%)
    - Keyword density (10%)
    """
    
    def __init__(self):
        """Initialize the hybrid scorer."""
        self.logger = logging.getLogger(__name__)
        self.config = config.scoring
        self.model_config = config.model
        
        # Initialize processing components
        self.text_processor = TextProcessor()
        self.similarity_calculator = SimilarityCalculator()
        
        # Initialize Groq LLM as core component
        self.groq_enhancer = GroqEnhancer(api_key=self.model_config.groq_api_key)
        if self.groq_enhancer.is_available():
            self.logger.info("Groq LLM integrated successfully")
        else:
            self.logger.warning("Groq LLM not available - using fallback explanations")
        
        # Scoring weights from config
        self.weights = {
            'skills': self.config.skills_weight,
            'experience': self.config.experience_weight,
            'education': self.config.education_weight,
            'keywords': self.config.keywords_weight
        }
        
        # Education level hierarchy for scoring
        self.education_hierarchy = {
            'phd': 6,
            'master': 5,
            'bachelor': 4,
            'associate': 3,
            'diploma': 2,
            'certificate': 1,
            'high_school': 0
        }
    
    def score_resume(
        self, 
        resume_data: ResumeData, 
        job_description: JobDescription
    ) -> ScoringResult:
        """
        Score a resume against a job description.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description data
            
        Returns:
            Complete scoring result with explanations
        """
        start_time = datetime.now()
        
        try:
            # Initialize score breakdown
            breakdown = ScoreBreakdown()
            
            # Calculate component scores
            breakdown.skills_score = self._score_skills(resume_data, job_description, breakdown)
            breakdown.experience_score = self._score_experience(resume_data, job_description, breakdown)
            breakdown.education_score = self._score_education(resume_data, job_description, breakdown)
            breakdown.keywords_score = self._score_keywords(resume_data, job_description, breakdown)
            
            # Calculate semantic similarity
            similarities = self.similarity_calculator.calculate_comprehensive_similarity(
                resume_data.get_all_text(),
                job_description.get_all_text(),
                resume_data.get_skill_names(),
                job_description.get_all_skills()
            )
            
            breakdown.semantic_similarity_score = similarities.get('semantic_similarity', 0.0)
            breakdown.keyword_matching_score = similarities.get('keyword_similarity', 0.0)
            
            # Calculate overall score using weighted average
            overall_score = (
                breakdown.skills_score * self.weights['skills'] +
                breakdown.experience_score * self.weights['experience'] +
                breakdown.education_score * self.weights['education'] +
                breakdown.keywords_score * self.weights['keywords']
            )
            
            # Apply semantic similarity boost/penalty
            semantic_weight = self.config.semantic_similarity_weight
            keyword_weight = self.config.keyword_matching_weight
            
            similarity_boost = (
                breakdown.semantic_similarity_score * semantic_weight +
                breakdown.keyword_matching_score * keyword_weight
            ) * 0.1  # 10% boost/penalty max
            
            overall_score = min(1.0, overall_score + similarity_boost)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(breakdown, similarities)
            
            # Generate base explanations
            explanation = self._generate_explanation(
                resume_data, 
                job_description, 
                breakdown, 
                overall_score
            )
            
            # Create scoring result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ScoringResult(
                overall_score=overall_score,
                confidence_score=confidence_score,
                breakdown=breakdown,
                explanation=explanation,
                processing_time_seconds=processing_time,
                resume_filename=resume_data.file_name,
                job_title=job_description.job_title
            )
            
            # Enhance with Groq LLM as core functionality
            if self.groq_enhancer.is_available():
                try:
                    enhanced_result = self.groq_enhancer.enhance_scoring_explanation(
                        result, resume_data, job_description
                    )
                    self.logger.info("Enhanced scoring with Groq LLM")
                    return enhanced_result
                except Exception as e:
                    self.logger.warning(f"Groq enhancement failed, using base result: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error scoring resume: {str(e)}")
            
            # Return minimal result on error
            return ScoringResult(
                overall_score=0.0,
                confidence_score=0.0,
                breakdown=ScoreBreakdown(),
                explanation=ScoreExplanation(
                    summary=f"Error during scoring: {str(e)}"
                ),
                resume_filename=resume_data.file_name,
                job_title=job_description.job_title
            )
    
    def _score_skills(
        self, 
        resume_data: ResumeData, 
        job_description: JobDescription,
        breakdown: ScoreBreakdown
    ) -> float:
        """Score skills matching between resume and job requirements."""
        try:
            resume_skills = resume_data.get_skill_names()
            job_required_skills = job_description.required_skills
            job_preferred_skills = job_description.preferred_skills
            
            if not job_required_skills and not job_preferred_skills:
                # If no specific skills listed, extract from text
                job_text_skills = self.text_processor.extract_skills_advanced(
                    job_description.get_all_text()
                )
                job_required_skills = [skill['name'] for skill in job_text_skills[:10]]
            
            # Perform skill gap analysis
            gap_analysis = self.similarity_calculator.calculate_skill_gap_analysis(
                resume_skills,
                job_required_skills,
                job_preferred_skills
            )
            
            # Update breakdown with skill details - ensure no overlap
            matched_skills = [skill for skill in gap_analysis['matched_required'] if skill]
            missing_skills = [skill for skill in gap_analysis['missing_required'] if skill]
            
            # Remove any skills that appear in both lists (shouldn't happen but safety check)
            matched_skills_clean = []
            missing_skills_clean = []
            
            for skill in matched_skills:
                if skill not in missing_skills:
                    matched_skills_clean.append(skill)
            
            for skill in missing_skills:
                if skill not in matched_skills:
                    missing_skills_clean.append(skill)
            
            breakdown.matched_skills = matched_skills_clean[:10]  # Top 10
            breakdown.missing_skills = missing_skills_clean[:10]  # Top 10
            breakdown.skill_match_percentage = gap_analysis['required_skill_coverage'] * 100
            
            # Calculate skill score
            required_score = gap_analysis['required_skill_coverage']
            preferred_score = gap_analysis['preferred_skill_coverage']
            
            # Weight required skills more heavily
            skill_score = (required_score * 0.8) + (preferred_score * 0.2)
            
            # Bonus for having many relevant skills
            if len(breakdown.matched_skills) >= 5:
                skill_score = min(1.0, skill_score + 0.1)
            
            return skill_score
            
        except Exception as e:
            self.logger.error(f"Error scoring skills: {str(e)}")
            return 0.0
    
    def _score_experience(
        self, 
        resume_data: ResumeData, 
        job_description: JobDescription,
        breakdown: ScoreBreakdown
    ) -> float:
        """Score experience relevance and years."""
        try:
            # Extract required experience from job description
            required_years = job_description.extract_years_experience()
            job_level = self.text_processor.identify_job_level(job_description.get_all_text())
            
            # Calculate candidate's total experience
            candidate_years = resume_data.get_years_experience()
            
            # If we can't extract years, estimate from experience entries
            if candidate_years == 0 and resume_data.experience:
                candidate_years = len(resume_data.experience) * 1.5  # Rough estimate
            
            # Base experience score
            experience_score = 0.0
            
            # Years experience scoring
            if required_years:
                if candidate_years >= required_years:
                    years_score = 1.0
                    # Bonus for significantly more experience
                    if candidate_years > required_years * 1.5:
                        years_score = min(1.0, years_score + 0.1)
                else:
                    # Partial credit for partial experience
                    years_score = candidate_years / required_years
                
                experience_score += years_score * 0.6
            else:
                # Default scoring based on job level
                level_requirements = {
                    'senior': 5,
                    'mid': 3,
                    'junior': 1,
                    'unknown': 2
                }
                
                required_for_level = level_requirements.get(job_level, 2)
                if candidate_years >= required_for_level:
                    experience_score += 0.6
                else:
                    experience_score += (candidate_years / required_for_level) * 0.6
            
            # Relevance of experience (based on job titles and descriptions)
            relevance_score = self._calculate_experience_relevance(
                resume_data.experience,
                job_description
            )
            experience_score += relevance_score * 0.4
            
            # Update breakdown
            breakdown.relevant_experience_years = candidate_years
            breakdown.required_experience_years = required_years or 0
            breakdown.experience_match_percentage = min(100, (candidate_years / max(required_years or 1, 1)) * 100)
            
            return min(1.0, experience_score)
            
        except Exception as e:
            self.logger.error(f"Error scoring experience: {str(e)}")
            return 0.0
    
    def _calculate_experience_relevance(
        self, 
        experiences: List, 
        job_description: JobDescription
    ) -> float:
        """Calculate how relevant the work experience is to the job."""
        if not experiences:
            return 0.0
        
        job_text = job_description.get_all_text().lower()
        job_title = job_description.job_title.lower()
        
        relevance_scores = []
        
        for exp in experiences:
            exp_score = 0.0
            
            # Job title similarity
            if exp.job_title:
                exp_title = exp.job_title.lower()
                title_similarity = self.text_processor.calculate_text_similarity(
                    exp_title, job_title
                )
                exp_score += title_similarity * 0.4
            
            # Description similarity
            if exp.description:
                desc_similarity = self.text_processor.calculate_text_similarity(
                    exp.description.lower(), job_text
                )
                exp_score += desc_similarity * 0.6
            
            relevance_scores.append(exp_score)
        
        # Return average relevance, weighted by position (recent experience more important)
        if relevance_scores:
            weights = [1.0 / (i + 1) for i in range(len(relevance_scores))]  # Decreasing weights
            weighted_score = sum(score * weight for score, weight in zip(relevance_scores, weights))
            total_weight = sum(weights)
            return weighted_score / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def _score_education(
        self, 
        resume_data: ResumeData, 
        job_description: JobDescription,
        breakdown: ScoreBreakdown
    ) -> float:
        """Score education alignment with job requirements."""
        try:
            if not resume_data.education:
                breakdown.education_level_match = False
                breakdown.education_field_match = False
                return 0.0
            
            # Extract education requirements from job description
            job_education_reqs = job_description.get_education_requirements()
            job_text = job_description.get_all_text().lower()
            
            # Get candidate's highest education
            highest_education = resume_data.get_highest_education()
            if not highest_education:
                return 0.0
            
            candidate_level = highest_education.get_degree_level()
            candidate_field = highest_education.field_of_study or ""
            
            education_score = 0.0
            
            # Level matching
            if job_education_reqs:
                # Extract required level from job requirements
                required_level = self._extract_required_education_level(job_education_reqs)
                
                candidate_level_rank = self.education_hierarchy.get(candidate_level, 0)
                required_level_rank = self.education_hierarchy.get(required_level, 0)
                
                if candidate_level_rank >= required_level_rank:
                    education_score += 0.7
                    breakdown.education_level_match = True
                else:
                    # Partial credit
                    education_score += (candidate_level_rank / required_level_rank) * 0.7
            else:
                # Default scoring - bachelor's degree gets good score
                if candidate_level in ['bachelor', 'master', 'phd']:
                    education_score += 0.7
                    breakdown.education_level_match = True
                elif candidate_level in ['associate', 'diploma']:
                    education_score += 0.5
                else:
                    education_score += 0.3
            
            # Field relevance
            if candidate_field:
                field_relevance = self._calculate_field_relevance(candidate_field, job_text)
                education_score += field_relevance * 0.3
                breakdown.education_field_match = field_relevance > 0.5
            
            return min(1.0, education_score)
            
        except Exception as e:
            self.logger.error(f"Error scoring education: {str(e)}")
            return 0.0
    
    def _extract_required_education_level(self, education_reqs: List[str]) -> str:
        """Extract the required education level from requirements."""
        combined_text = " ".join(education_reqs).lower()
        
        # Check for degree requirements in order of preference
        if any(term in combined_text for term in ['phd', 'ph.d', 'doctorate']):
            return 'phd'
        elif any(term in combined_text for term in ['master', 'mba', 'm.s', 'm.a']):
            return 'master'
        elif any(term in combined_text for term in ['bachelor', 'b.s', 'b.a', 'degree']):
            return 'bachelor'
        elif any(term in combined_text for term in ['associate', 'diploma']):
            return 'associate'
        else:
            return 'bachelor'  # Default assumption
    
    def _calculate_field_relevance(self, candidate_field: str, job_text: str) -> float:
        """Calculate relevance of education field to job requirements."""
        # Define field mappings
        field_mappings = {
            'computer science': ['software', 'programming', 'development', 'tech', 'it'],
            'engineering': ['engineering', 'technical', 'development', 'design'],
            'data science': ['data', 'analytics', 'statistics', 'machine learning', 'ai'],
            'business': ['business', 'management', 'finance', 'marketing', 'sales'],
            'mathematics': ['math', 'statistics', 'analytics', 'quantitative'],
            'physics': ['physics', 'research', 'analysis', 'modeling']
        }
        
        candidate_field_lower = candidate_field.lower()
        
        # Direct field mention in job text
        if candidate_field_lower in job_text:
            return 1.0
        
        # Check field mappings
        for field, keywords in field_mappings.items():
            if field in candidate_field_lower:
                for keyword in keywords:
                    if keyword in job_text:
                        return 0.8
        
        # Semantic similarity as fallback
        return self.text_processor.calculate_text_similarity(candidate_field, job_text)
    
    def _score_keywords(
        self, 
        resume_data: ResumeData, 
        job_description: JobDescription,
        breakdown: ScoreBreakdown
    ) -> float:
        """Score keyword matching and density."""
        try:
            resume_text = resume_data.get_all_text()
            job_text = job_description.get_all_text()
            
            # Extract keywords from job description
            job_keywords = self.text_processor.extract_keywords(job_text, top_k=30)
            job_keyword_names = [kw['keyword'] for kw in job_keywords]
            
            # Find matching keywords
            matching_keywords = self.similarity_calculator.find_matching_keywords(
                resume_text, job_text, top_k=20
            )
            
            # Update breakdown
            breakdown.matched_keywords = [kw['keyword'] for kw in matching_keywords]
            breakdown.total_keywords = len(job_keyword_names)
            
            # Calculate keyword density in resume
            keyword_densities = self.text_processor.calculate_keyword_density(
                resume_text, job_keyword_names
            )
            
            # Average density
            if keyword_densities:
                avg_density = sum(keyword_densities.values()) / len(keyword_densities)
                breakdown.keyword_density = avg_density
            
            # Score based on matches and density
            if job_keyword_names:
                match_ratio = len(breakdown.matched_keywords) / len(job_keyword_names)
                density_score = min(1.0, breakdown.keyword_density * 100)  # Scale density
                
                keyword_score = (match_ratio * 0.7) + (density_score * 0.3)
            else:
                keyword_score = 0.5  # Default score if no keywords extracted
            
            return min(1.0, keyword_score)
            
        except Exception as e:
            self.logger.error(f"Error scoring keywords: {str(e)}")
            return 0.0
    
    def _calculate_confidence(
        self, 
        breakdown: ScoreBreakdown, 
        similarities: Dict[str, float]
    ) -> float:
        """Calculate confidence score for the overall assessment."""
        confidence_factors = []
        
        # Data completeness factor
        data_completeness = 0.0
        if breakdown.matched_skills:
            data_completeness += 0.25
        if breakdown.relevant_experience_years > 0:
            data_completeness += 0.25
        if breakdown.education_level_match:
            data_completeness += 0.25
        if breakdown.matched_keywords:
            data_completeness += 0.25
        
        confidence_factors.append(data_completeness)
        
        # Similarity consistency factor
        sim_scores = [
            similarities.get('tfidf_similarity', 0),
            similarities.get('semantic_similarity', 0),
            similarities.get('keyword_similarity', 0)
        ]
        
        valid_scores = [s for s in sim_scores if s > 0]
        if len(valid_scores) > 1:
            # Check consistency (low variance = high confidence)
            import statistics
            variance = statistics.variance(valid_scores)
            consistency = 1.0 / (1.0 + variance * 5)  # Scale variance
            confidence_factors.append(consistency)
        
        # Score magnitude factor (higher scores generally more reliable)
        score_components = [
            breakdown.skills_score,
            breakdown.experience_score,
            breakdown.education_score,
            breakdown.keywords_score
        ]
        
        avg_score = sum(score_components) / len(score_components)
        magnitude_confidence = min(1.0, avg_score * 1.2)
        confidence_factors.append(magnitude_confidence)
        
        # Text length factor (more text = more reliable analysis)
        text_length_factor = min(1.0, similarities.get('confidence', 0.5))
        confidence_factors.append(text_length_factor)
        
        # Overall confidence
        if confidence_factors:
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            return min(1.0, overall_confidence)
        
        return 0.5  # Default moderate confidence
    
    def _generate_explanation(
        self,
        resume_data: ResumeData,
        job_description: JobDescription,
        breakdown: ScoreBreakdown,
        overall_score: float
    ) -> ScoreExplanation:
        """Generate human-readable explanation of the scoring."""
        explanation = ScoreExplanation()
        
        try:
            # Overall summary
            score_category = self._get_score_category(overall_score)
            explanation.summary = (
                f"This resume shows a {score_category.lower()} with the job requirements, "
                f"achieving an overall relevance score of {int(overall_score * 100)}%. "
            )
            
            # Skills explanation
            if breakdown.matched_skills:
                skill_count = len(breakdown.matched_skills)
                explanation.skills_explanation = (
                    f"Strong skill alignment with {skill_count} matched skills including "
                    f"{', '.join(breakdown.matched_skills[:3])}. "
                )
            else:
                explanation.skills_explanation = "Limited skill matching identified. "
            
            if breakdown.missing_skills:
                explanation.skills_explanation += (
                    f"Key missing skills: {', '.join(breakdown.missing_skills[:3])}."
                )
            
            # Experience explanation
            if breakdown.relevant_experience_years > 0:
                explanation.experience_explanation = (
                    f"Candidate has {breakdown.relevant_experience_years:.1f} years of "
                    f"relevant experience"
                )
                if breakdown.required_experience_years:
                    explanation.experience_explanation += (
                        f" against {breakdown.required_experience_years} years required"
                    )
                explanation.experience_explanation += ". "
            else:
                explanation.experience_explanation = "Experience information limited or not clearly relevant. "
            
            # Education explanation
            if breakdown.education_level_match:
                explanation.education_explanation = "Educational background aligns well with job requirements. "
            else:
                explanation.education_explanation = "Educational background may not fully meet requirements. "
            
            if breakdown.education_field_match:
                explanation.education_explanation += "Field of study is relevant to the role."
            
            # Generate strengths and weaknesses
            explanation.strengths = self._identify_strengths(breakdown, resume_data)
            explanation.weaknesses = self._identify_weaknesses(breakdown, job_description)
            explanation.recommendations = self._generate_recommendations(breakdown, job_description)
            
            # Skill gaps
            explanation.skill_gaps = breakdown.missing_skills[:5]
            
            # Experience gaps
            if breakdown.required_experience_years > breakdown.relevant_experience_years:
                years_gap = breakdown.required_experience_years - breakdown.relevant_experience_years
                explanation.experience_gaps = [f"Need {years_gap:.1f} more years of experience"]
            
            # Improvement suggestions
            explanation.improvement_suggestions = self._generate_improvement_suggestions(breakdown)
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            explanation.summary = "Unable to generate detailed explanation due to processing error."
        
        return explanation
    
    def _get_score_category(self, score: float) -> str:
        """Get human-readable score category."""
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
        else:
            return "Poor match"
    
    def _identify_strengths(self, breakdown: ScoreBreakdown, resume_data: ResumeData) -> List[str]:
        """Identify candidate strengths."""
        strengths = []
        
        if breakdown.skills_score >= 0.8:
            strengths.append("Strong technical skill set")
        
        if breakdown.experience_score >= 0.8:
            strengths.append("Relevant work experience")
        
        if breakdown.education_score >= 0.8:
            strengths.append("Strong educational background")
        
        if len(breakdown.matched_skills) >= 5:
            strengths.append("Diverse skill portfolio")
        
        if breakdown.keyword_density > 0.05:
            strengths.append("Good keyword alignment")
        
        if len(resume_data.experience) >= 3:
            strengths.append("Extensive work history")
        
        if resume_data.certifications:
            strengths.append("Professional certifications")
        
        if not strengths:
            strengths.append("Shows potential for the role")
        
        return strengths[:5]  # Limit to top 5
    
    def _identify_weaknesses(self, breakdown: ScoreBreakdown, job_description: JobDescription) -> List[str]:
        """Identify candidate weaknesses."""
        weaknesses = []
        
        if breakdown.skills_score < 0.5:
            weaknesses.append("Limited skill matching")
        
        if breakdown.experience_score < 0.5:
            weaknesses.append("Insufficient relevant experience")
        
        if breakdown.education_score < 0.5:
            weaknesses.append("Educational background below requirements")
        
        if len(breakdown.missing_skills) >= 3:
            weaknesses.append("Missing key technical skills")
        
        if breakdown.keyword_density < 0.02:
            weaknesses.append("Limited keyword relevance")
        
        if breakdown.required_experience_years and breakdown.relevant_experience_years < breakdown.required_experience_years * 0.7:
            weaknesses.append("Below required experience level")
        
        if not weaknesses:
            weaknesses.append("Minor areas for improvement")
        
        return weaknesses[:5]  # Limit to top 5
    
    def _generate_recommendations(self, breakdown: ScoreBreakdown, job_description: JobDescription) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if breakdown.missing_skills:
            recommendations.append(f"Develop skills in: {', '.join(breakdown.missing_skills[:3])}")
        
        if breakdown.experience_score < 0.7:
            recommendations.append("Gain more relevant work experience")
        
        if breakdown.keyword_density < 0.03:
            recommendations.append("Include more job-relevant keywords in resume")
        
        if breakdown.education_score < 0.6:
            recommendations.append("Consider additional certifications or training")
        
        if len(breakdown.matched_skills) < 3:
            recommendations.append("Highlight transferable skills more prominently")
        
        if not recommendations:
            recommendations.append("Continue building on current strengths")
        
        return recommendations[:5]
    
    def _generate_improvement_suggestions(self, breakdown: ScoreBreakdown) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        if breakdown.skills_score < 0.6:
            suggestions.append("Take online courses in missing technical skills")
        
        if breakdown.experience_score < 0.6:
            suggestions.append("Seek projects or internships in relevant areas")
        
        if breakdown.keyword_density < 0.02:
            suggestions.append("Optimize resume with industry-specific terminology")
        
        if not breakdown.education_level_match:
            suggestions.append("Consider pursuing additional qualifications")
        
        if len(breakdown.matched_keywords) < 5:
            suggestions.append("Research and include relevant industry keywords")
        
        return suggestions[:3]  # Keep focused
    
    def batch_score_resumes(
        self,
        resume_list: List[ResumeData],
        job_description: JobDescription
    ) -> List[ScoringResult]:
        """
        Score multiple resumes against a job description.
        
        Args:
            resume_list: List of parsed resume data
            job_description: Job description to score against
            
        Returns:
            List of scoring results sorted by score
        """
        results = []
        
        for resume_data in resume_list:
            try:
                result = self.score_resume(resume_data, job_description)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error scoring resume {resume_data.file_name}: {str(e)}")
                # Add error result
                error_result = ScoringResult(
                    overall_score=0.0,
                    confidence_score=0.0,
                    breakdown=ScoreBreakdown(),
                    explanation=ScoreExplanation(summary=f"Scoring error: {str(e)}"),
                    resume_filename=resume_data.file_name,
                    job_title=job_description.job_title
                )
                results.append(error_result)
        
        # Sort by overall score (descending)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return results

