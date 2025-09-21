"""
Data models for multi-resume comparison and ranking results.

This module contains classes for representing comparison results when
evaluating multiple resumes against a single job description.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from .scoring_result import ScoringResult


@dataclass
class RankingResult:
    """Individual resume ranking result."""
    
    # Resume identification
    resume_filename: str
    resume_id: Optional[str] = None
    candidate_name: Optional[str] = None
    
    # Scoring information
    scoring_result: ScoringResult = field(default_factory=ScoringResult)
    
    # Ranking information
    rank: int = 0
    percentile: float = 0.0
    
    # Relative comparison
    score_vs_average: float = 0.0  # How much above/below average
    score_vs_best: float = 0.0     # How much below the best score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resume_filename": self.resume_filename,
            "resume_id": self.resume_id,
            "candidate_name": self.candidate_name,
            "scoring_result": self.scoring_result.to_dict(),
            "rank": self.rank,
            "percentile": self.percentile,
            "score_vs_average": self.score_vs_average,
            "score_vs_best": self.score_vs_best
        }
    
    @property
    def overall_score(self) -> float:
        """Get overall score from scoring result."""
        return self.scoring_result.overall_score
    
    @property
    def confidence_score(self) -> float:
        """Get confidence score from scoring result."""
        return self.scoring_result.confidence_score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary information for display."""
        return {
            "filename": self.resume_filename,
            "candidate": self.candidate_name or "Unknown",
            "rank": self.rank,
            "score": self.overall_score,
            "score_percentage": int(self.overall_score * 100),
            "confidence": self.confidence_score,
            "category": self.scoring_result.get_score_category(),
            "percentile": self.percentile
        }


@dataclass
class ComparisonResult:
    """Complete comparison result for multiple resumes."""
    
    # Job information
    job_title: str = ""
    job_description_summary: str = ""
    
    # Ranking results
    rankings: List[RankingResult] = field(default_factory=list)
    
    # Statistical information
    total_resumes: int = 0
    average_score: float = 0.0
    median_score: float = 0.0
    best_score: float = 0.0
    worst_score: float = 0.0
    score_std_dev: float = 0.0
    
    # Score distribution
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Analysis insights
    common_strengths: List[str] = field(default_factory=list)
    common_weaknesses: List[str] = field(default_factory=list)
    skill_gaps_analysis: Dict[str, int] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    created_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_timestamp is None:
            self.created_timestamp = datetime.now()
        
        self.total_resumes = len(self.rankings)
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate statistical measures from rankings."""
        if not self.rankings:
            return
        
        scores = [ranking.overall_score for ranking in self.rankings]
        
        # Basic statistics
        self.average_score = sum(scores) / len(scores)
        self.best_score = max(scores)
        self.worst_score = min(scores)
        
        # Median calculation
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            self.median_score = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            self.median_score = sorted_scores[n//2]
        
        # Standard deviation
        if len(scores) > 1:
            variance = sum((score - self.average_score) ** 2 for score in scores) / len(scores)
            self.score_std_dev = variance ** 0.5
        
        # Score distribution
        self.score_distribution = {
            "Excellent (90-100%)": len([s for s in scores if s >= 0.9]),
            "Very Good (80-89%)": len([s for s in scores if 0.8 <= s < 0.9]),
            "Good (70-79%)": len([s for s in scores if 0.7 <= s < 0.8]),
            "Fair (60-69%)": len([s for s in scores if 0.6 <= s < 0.7]),
            "Moderate (50-59%)": len([s for s in scores if 0.5 <= s < 0.6]),
            "Poor (40-49%)": len([s for s in scores if 0.4 <= s < 0.5]),
            "Very Poor (<40%)": len([s for s in scores if s < 0.4])
        }
        
        # Update relative scores
        for ranking in self.rankings:
            ranking.score_vs_average = ranking.overall_score - self.average_score
            ranking.score_vs_best = ranking.overall_score - self.best_score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison result to dictionary representation.
        
        Returns:
            Dictionary containing all comparison result data
        """
        return {
            "job_title": self.job_title,
            "job_description_summary": self.job_description_summary,
            "rankings": [ranking.to_dict() for ranking in self.rankings],
            "total_resumes": self.total_resumes,
            "average_score": self.average_score,
            "median_score": self.median_score,
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "score_std_dev": self.score_std_dev,
            "score_distribution": self.score_distribution,
            "common_strengths": self.common_strengths,
            "common_weaknesses": self.common_weaknesses,
            "skill_gaps_analysis": self.skill_gaps_analysis,
            "processing_time_seconds": self.processing_time_seconds,
            "created_timestamp": self.created_timestamp.isoformat() if self.created_timestamp else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert comparison result to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResult':
        """
        Create ComparisonResult instance from dictionary.
        
        Args:
            data: Dictionary containing comparison result data
            
        Returns:
            ComparisonResult instance
        """
        rankings = []
        for ranking_data in data.get("rankings", []):
            scoring_result = ScoringResult.from_dict(ranking_data.get("scoring_result", {}))
            ranking = RankingResult(
                resume_filename=ranking_data.get("resume_filename", ""),
                resume_id=ranking_data.get("resume_id"),
                candidate_name=ranking_data.get("candidate_name"),
                scoring_result=scoring_result,
                rank=ranking_data.get("rank", 0),
                percentile=ranking_data.get("percentile", 0.0),
                score_vs_average=ranking_data.get("score_vs_average", 0.0),
                score_vs_best=ranking_data.get("score_vs_best", 0.0)
            )
            rankings.append(ranking)
        
        created_timestamp = None
        if data.get("created_timestamp"):
            try:
                created_timestamp = datetime.fromisoformat(data["created_timestamp"])
            except ValueError:
                pass
        
        return cls(
            job_title=data.get("job_title", ""),
            job_description_summary=data.get("job_description_summary", ""),
            rankings=rankings,
            total_resumes=data.get("total_resumes", 0),
            average_score=data.get("average_score", 0.0),
            median_score=data.get("median_score", 0.0),
            best_score=data.get("best_score", 0.0),
            worst_score=data.get("worst_score", 0.0),
            score_std_dev=data.get("score_std_dev", 0.0),
            score_distribution=data.get("score_distribution", {}),
            common_strengths=data.get("common_strengths", []),
            common_weaknesses=data.get("common_weaknesses", []),
            skill_gaps_analysis=data.get("skill_gaps_analysis", {}),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            created_timestamp=created_timestamp
        )
    
    def add_ranking(self, ranking: RankingResult):
        """
        Add a ranking result and recalculate statistics.
        
        Args:
            ranking: RankingResult to add
        """
        self.rankings.append(ranking)
        self.total_resumes = len(self.rankings)
        self._calculate_statistics()
        self._update_ranks_and_percentiles()
    
    def _update_ranks_and_percentiles(self):
        """Update ranks and percentiles for all rankings."""
        # Sort by score (descending)
        sorted_rankings = sorted(self.rankings, key=lambda x: x.overall_score, reverse=True)
        
        # Update ranks
        for i, ranking in enumerate(sorted_rankings):
            ranking.rank = i + 1
            ranking.percentile = ((self.total_resumes - i) / self.total_resumes) * 100
        
        # Update the original rankings list to maintain the sorted order
        self.rankings = sorted_rankings
    
    def get_top_candidates(self, n: int = 5) -> List[RankingResult]:
        """
        Get top N candidates by score.
        
        Args:
            n: Number of top candidates to return
            
        Returns:
            List of top ranking results
        """
        return self.rankings[:n]
    
    def get_candidates_above_threshold(self, threshold: float = 0.7) -> List[RankingResult]:
        """
        Get candidates above a score threshold.
        
        Args:
            threshold: Minimum score threshold
            
        Returns:
            List of qualifying ranking results
        """
        return [ranking for ranking in self.rankings if ranking.overall_score >= threshold]
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get statistical summary of the comparison.
        
        Returns:
            Dictionary with statistical information
        """
        return {
            "total_resumes": self.total_resumes,
            "score_statistics": {
                "average": round(self.average_score, 3),
                "median": round(self.median_score, 3),
                "best": round(self.best_score, 3),
                "worst": round(self.worst_score, 3),
                "std_dev": round(self.score_std_dev, 3)
            },
            "score_distribution": self.score_distribution,
            "processing_time": round(self.processing_time_seconds, 2)
        }
    
    def analyze_skill_gaps(self):
        """Analyze common skill gaps across all resumes."""
        skill_mentions = {}
        total_resumes = len(self.rankings)
        
        for ranking in self.rankings:
            missing_skills = ranking.scoring_result.breakdown.missing_skills
            for skill in missing_skills:
                skill_mentions[skill] = skill_mentions.get(skill, 0) + 1
        
        # Convert to percentages and sort by frequency
        self.skill_gaps_analysis = {
            skill: round((count / total_resumes) * 100, 1)
            for skill, count in sorted(skill_mentions.items(), 
                                     key=lambda x: x[1], reverse=True)
        }
    
    def analyze_common_patterns(self):
        """Analyze common strengths and weaknesses."""
        all_strengths = []
        all_weaknesses = []
        
        for ranking in self.rankings:
            all_strengths.extend(ranking.scoring_result.explanation.strengths)
            all_weaknesses.extend(ranking.scoring_result.explanation.weaknesses)
        
        # Count frequency and get most common
        strength_counts = {}
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        # Get top 5 most common
        self.common_strengths = [
            strength for strength, _ in 
            sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        self.common_weaknesses = [
            weakness for weakness, _ in 
            sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def export_summary_table(self) -> List[Dict[str, Any]]:
        """
        Export comparison results as a table for CSV/Excel export.
        
        Returns:
            List of dictionaries for table export
        """
        table_data = []
        
        for ranking in self.rankings:
            row = {
                "Rank": ranking.rank,
                "Candidate": ranking.candidate_name or "Unknown",
                "Filename": ranking.resume_filename,
                "Overall Score": f"{ranking.overall_score:.3f}",
                "Score Percentage": f"{int(ranking.overall_score * 100)}%",
                "Score Category": ranking.scoring_result.get_score_category(),
                "Skills Score": f"{ranking.scoring_result.breakdown.skills_score:.3f}",
                "Experience Score": f"{ranking.scoring_result.breakdown.experience_score:.3f}",
                "Education Score": f"{ranking.scoring_result.breakdown.education_score:.3f}",
                "Keywords Score": f"{ranking.scoring_result.breakdown.keywords_score:.3f}",
                "Confidence": f"{ranking.confidence_score:.3f}",
                "Percentile": f"{ranking.percentile:.1f}%",
                "vs Average": f"{ranking.score_vs_average:+.3f}",
                "vs Best": f"{ranking.score_vs_best:+.3f}",
                "Matched Skills": ", ".join(ranking.scoring_result.breakdown.matched_skills[:5]),
                "Missing Skills": ", ".join(ranking.scoring_result.breakdown.missing_skills[:5])
            }
            table_data.append(row)
        
        return table_data

