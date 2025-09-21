"""
Data models and classes for the Resume Relevance Check System.

This package contains all data models, classes, and type definitions used
throughout the application.
"""

from .resume_data import ResumeData, ContactInfo, Education, Experience, Skill
from .job_description import JobDescription, JobRequirement
from .scoring_result import ScoringResult, ScoreBreakdown, ScoreExplanation
from .comparison_result import ComparisonResult, RankingResult

__all__ = [
    "ResumeData",
    "ContactInfo", 
    "Education",
    "Experience",
    "Skill",
    "JobDescription",
    "JobRequirement",
    "ScoringResult",
    "ScoreBreakdown",
    "ScoreExplanation",
    "ComparisonResult",
    "RankingResult"
]

