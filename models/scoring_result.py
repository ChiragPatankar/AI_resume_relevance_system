"""
Data models for scoring results and explanations.

This module contains classes for representing scoring results, breakdowns,
and explanations from the resume relevance evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of scoring components."""
    
    # Main category scores (0.0 to 1.0)
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    keywords_score: float = 0.0
    
    # Sub-component scores
    semantic_similarity_score: float = 0.0
    keyword_matching_score: float = 0.0
    
    # Detailed skill analysis
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    skill_match_percentage: float = 0.0
    
    # Experience analysis
    relevant_experience_years: float = 0.0
    required_experience_years: float = 0.0
    experience_match_percentage: float = 0.0
    
    # Education analysis
    education_level_match: bool = False
    education_field_match: bool = False
    
    # Keyword analysis
    matched_keywords: List[str] = field(default_factory=list)
    total_keywords: int = 0
    keyword_density: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "skills_score": self.skills_score,
            "experience_score": self.experience_score,
            "education_score": self.education_score,
            "keywords_score": self.keywords_score,
            "semantic_similarity_score": self.semantic_similarity_score,
            "keyword_matching_score": self.keyword_matching_score,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "skill_match_percentage": self.skill_match_percentage,
            "relevant_experience_years": self.relevant_experience_years,
            "required_experience_years": self.required_experience_years,
            "experience_match_percentage": self.experience_match_percentage,
            "education_level_match": self.education_level_match,
            "education_field_match": self.education_field_match,
            "matched_keywords": self.matched_keywords,
            "total_keywords": self.total_keywords,
            "keyword_density": self.keyword_density
        }


@dataclass
class ScoreExplanation:
    """Human-readable explanation of the scoring."""
    
    # Overall summary
    summary: str = ""
    
    # Category explanations
    skills_explanation: str = ""
    experience_explanation: str = ""
    education_explanation: str = ""
    
    # Strengths and weaknesses
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Detailed insights
    skill_gaps: List[str] = field(default_factory=list)
    experience_gaps: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "summary": self.summary,
            "skills_explanation": self.skills_explanation,
            "experience_explanation": self.experience_explanation,
            "education_explanation": self.education_explanation,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "skill_gaps": self.skill_gaps,
            "experience_gaps": self.experience_gaps,
            "improvement_suggestions": self.improvement_suggestions
        }


@dataclass
class ScoringResult:
    """Complete scoring result for a resume against a job description."""
    
    # Overall score and confidence
    overall_score: float = 0.0
    confidence_score: float = 0.0
    fit_level: str = "Low"
    
    # Score breakdown
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    
    # Explanation
    explanation: ScoreExplanation = field(default_factory=ScoreExplanation)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    model_version: str = "1.0"
    scoring_method: str = "hybrid"
    
    # Input references
    resume_filename: Optional[str] = None
    job_title: Optional[str] = None
    
    # Timestamp
    created_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_timestamp is None:
            self.created_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scoring result to dictionary representation.
        
        Returns:
            Dictionary containing all scoring result data
        """
        return {
            "overall_score": self.overall_score,
            "confidence_score": self.confidence_score,
            "fit_level": self.fit_level,
            "breakdown": self.breakdown.to_dict(),
            "explanation": self.explanation.to_dict(),
            "processing_time_seconds": self.processing_time_seconds,
            "model_version": self.model_version,
            "scoring_method": self.scoring_method,
            "resume_filename": self.resume_filename,
            "job_title": self.job_title,
            "created_timestamp": self.created_timestamp.isoformat() if self.created_timestamp else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert scoring result to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoringResult':
        """
        Create ScoringResult instance from dictionary.
        
        Args:
            data: Dictionary containing scoring result data
            
        Returns:
            ScoringResult instance
        """
        breakdown = ScoreBreakdown(**data.get("breakdown", {}))
        explanation = ScoreExplanation(**data.get("explanation", {}))
        
        created_timestamp = None
        if data.get("created_timestamp"):
            try:
                created_timestamp = datetime.fromisoformat(data["created_timestamp"])
            except ValueError:
                pass
        
        return cls(
            overall_score=data.get("overall_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            breakdown=breakdown,
            explanation=explanation,
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            model_version=data.get("model_version", "1.0"),
            scoring_method=data.get("scoring_method", "hybrid"),
            resume_filename=data.get("resume_filename"),
            job_title=data.get("job_title"),
            created_timestamp=created_timestamp
        )
    
    def get_score_category(self) -> str:
        """
        Get human-readable score category.
        
        Returns:
            Score category string
        """
        if self.overall_score >= 0.9:
            return "Excellent Match"
        elif self.overall_score >= 0.8:
            return "Very Good Match"
        elif self.overall_score >= 0.7:
            return "Good Match"
        elif self.overall_score >= 0.6:
            return "Fair Match"
        elif self.overall_score >= 0.5:
            return "Moderate Match"
        elif self.overall_score >= 0.4:
            return "Poor Match"
        else:
            return "Very Poor Match"
    
    def get_score_percentage(self) -> int:
        """
        Get score as percentage.
        
        Returns:
            Score as integer percentage
        """
        return int(self.overall_score * 100)
    
    def get_confidence_category(self) -> str:
        """
        Get human-readable confidence category.
        
        Returns:
            Confidence category string
        """
        if self.confidence_score >= 0.9:
            return "Very High"
        elif self.confidence_score >= 0.8:
            return "High"
        elif self.confidence_score >= 0.7:
            return "Medium"
        elif self.confidence_score >= 0.6:
            return "Low"
        else:
            return "Very Low"
    
    def get_top_strengths(self, n: int = 3) -> List[str]:
        """
        Get top N strengths.
        
        Args:
            n: Number of strengths to return
            
        Returns:
            List of top strengths
        """
        return self.explanation.strengths[:n]
    
    def get_top_weaknesses(self, n: int = 3) -> List[str]:
        """
        Get top N weaknesses.
        
        Args:
            n: Number of weaknesses to return
            
        Returns:
            List of top weaknesses
        """
        return self.explanation.weaknesses[:n]
    
    def get_score_components(self) -> Dict[str, float]:
        """
        Get all score components as a dictionary.
        
        Returns:
            Dictionary mapping component names to scores
        """
        return {
            "Skills": self.breakdown.skills_score,
            "Experience": self.breakdown.experience_score,
            "Education": self.breakdown.education_score,
            "Keywords": self.breakdown.keywords_score
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis summary.
        
        Returns:
            Dictionary with detailed analysis
        """
        return {
            "overall_assessment": {
                "score": self.overall_score,
                "percentage": self.get_score_percentage(),
                "category": self.get_score_category(),
                "confidence": self.confidence_score,
                "confidence_category": self.get_confidence_category()
            },
            "component_scores": self.get_score_components(),
            "skills_analysis": {
                "matched": self.breakdown.matched_skills,
                "missing": self.breakdown.missing_skills,
                "match_percentage": self.breakdown.skill_match_percentage
            },
            "experience_analysis": {
                "relevant_years": self.breakdown.relevant_experience_years,
                "required_years": self.breakdown.required_experience_years,
                "match_percentage": self.breakdown.experience_match_percentage
            },
            "education_analysis": {
                "level_match": self.breakdown.education_level_match,
                "field_match": self.breakdown.education_field_match
            },
            "keywords_analysis": {
                "matched": self.breakdown.matched_keywords,
                "total": self.breakdown.total_keywords,
                "density": self.breakdown.keyword_density
            },
            "insights": {
                "strengths": self.explanation.strengths,
                "weaknesses": self.explanation.weaknesses,
                "recommendations": self.explanation.recommendations
            }
        }
    
    def is_good_match(self, threshold: float = 0.7) -> bool:
        """
        Check if this is considered a good match.
        
        Args:
            threshold: Minimum score to be considered good match
            
        Returns:
            True if score is above threshold
        """
        return self.overall_score >= threshold
    
    def has_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Check if the scoring has high confidence.
        
        Args:
            threshold: Minimum confidence to be considered high
            
        Returns:
            True if confidence is above threshold
        """
        return self.confidence_score >= threshold

