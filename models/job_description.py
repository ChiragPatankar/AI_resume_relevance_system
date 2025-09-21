"""
Data models for job description information.

This module contains classes for structured job description data representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class JobRequirement:
    """Individual job requirement or qualification."""
    
    text: str
    category: str  # "required", "preferred", "nice_to_have"
    requirement_type: str  # "skill", "experience", "education", "other"
    importance_weight: float = 1.0
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "category": self.category,
            "requirement_type": self.requirement_type,
            "importance_weight": self.importance_weight,
            "keywords": self.keywords
        }


@dataclass
class JobDescription:
    """Main class for structured job description information."""
    
    # Basic job information
    job_title: str = ""
    company: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None  # "full-time", "part-time", "contract", etc.
    remote_option: Optional[str] = None  # "remote", "hybrid", "on-site"
    
    # Job details
    summary: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    
    # Requirements
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    required_experience: List[str] = field(default_factory=list)
    preferred_experience: List[str] = field(default_factory=list)
    education_requirements: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    
    # Structured requirements
    requirements: List[JobRequirement] = field(default_factory=list)
    
    # Additional information
    salary_range: Optional[str] = None
    benefits: List[str] = field(default_factory=list)
    company_culture: Optional[str] = None
    
    # Processing information
    raw_text: str = ""
    processed_text: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Metadata
    source: Optional[str] = None
    posting_date: Optional[datetime] = None
    created_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_timestamp is None:
            self.created_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job description to dictionary representation.
        
        Returns:
            Dictionary containing all job description data
        """
        return {
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "job_type": self.job_type,
            "remote_option": self.remote_option,
            "summary": self.summary,
            "description": self.description,
            "responsibilities": self.responsibilities,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "required_experience": self.required_experience,
            "preferred_experience": self.preferred_experience,
            "education_requirements": self.education_requirements,
            "certifications": self.certifications,
            "requirements": [req.to_dict() for req in self.requirements],
            "salary_range": self.salary_range,
            "benefits": self.benefits,
            "company_culture": self.company_culture,
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "keywords": self.keywords,
            "source": self.source,
            "posting_date": self.posting_date.isoformat() if self.posting_date else None,
            "created_timestamp": self.created_timestamp.isoformat() if self.created_timestamp else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert job description to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobDescription':
        """
        Create JobDescription instance from dictionary.
        
        Args:
            data: Dictionary containing job description data
            
        Returns:
            JobDescription instance
        """
        requirements = [
            JobRequirement(**req_data) 
            for req_data in data.get("requirements", [])
        ]
        
        posting_date = None
        if data.get("posting_date"):
            try:
                posting_date = datetime.fromisoformat(data["posting_date"])
            except ValueError:
                pass
        
        created_timestamp = None
        if data.get("created_timestamp"):
            try:
                created_timestamp = datetime.fromisoformat(data["created_timestamp"])
            except ValueError:
                pass
        
        return cls(
            job_title=data.get("job_title", ""),
            company=data.get("company"),
            location=data.get("location"),
            job_type=data.get("job_type"),
            remote_option=data.get("remote_option"),
            summary=data.get("summary"),
            description=data.get("description"),
            responsibilities=data.get("responsibilities", []),
            required_skills=data.get("required_skills", []),
            preferred_skills=data.get("preferred_skills", []),
            required_experience=data.get("required_experience", []),
            preferred_experience=data.get("preferred_experience", []),
            education_requirements=data.get("education_requirements", []),
            certifications=data.get("certifications", []),
            requirements=requirements,
            salary_range=data.get("salary_range"),
            benefits=data.get("benefits", []),
            company_culture=data.get("company_culture"),
            raw_text=data.get("raw_text", ""),
            processed_text=data.get("processed_text", ""),
            keywords=data.get("keywords", []),
            source=data.get("source"),
            posting_date=posting_date,
            created_timestamp=created_timestamp
        )
    
    @classmethod
    def from_text(cls, text: str, job_title: str = "", company: str = "") -> 'JobDescription':
        """
        Create JobDescription instance from raw text.
        
        Args:
            text: Raw job description text
            job_title: Optional job title
            company: Optional company name
            
        Returns:
            JobDescription instance
        """
        return cls(
            job_title=job_title,
            company=company,
            description=text,
            raw_text=text
        )
    
    def get_all_text(self) -> str:
        """
        Get all textual content from the job description.
        
        Returns:
            Combined text from all sections
        """
        text_parts = []
        
        if self.job_title:
            text_parts.append(self.job_title)
        
        if self.summary:
            text_parts.append(self.summary)
        
        if self.description:
            text_parts.append(self.description)
        
        text_parts.extend(self.responsibilities)
        text_parts.extend(self.required_skills)
        text_parts.extend(self.preferred_skills)
        text_parts.extend(self.required_experience)
        text_parts.extend(self.preferred_experience)
        text_parts.extend(self.education_requirements)
        text_parts.extend(self.certifications)
        text_parts.extend(self.benefits)
        
        if self.company_culture:
            text_parts.append(self.company_culture)
        
        # Add requirement texts
        for req in self.requirements:
            text_parts.append(req.text)
        
        return " ".join(text_parts)
    
    def get_all_skills(self) -> List[str]:
        """
        Get all skills mentioned in the job description.
        
        Returns:
            Combined list of required and preferred skills
        """
        all_skills = []
        all_skills.extend(self.required_skills)
        all_skills.extend(self.preferred_skills)
        
        # Extract skills from requirements
        for req in self.requirements:
            if req.requirement_type == "skill":
                all_skills.extend(req.keywords)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in all_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        return unique_skills
    
    def get_experience_requirements(self) -> List[str]:
        """
        Get all experience requirements.
        
        Returns:
            Combined list of required and preferred experience
        """
        experience = []
        experience.extend(self.required_experience)
        experience.extend(self.preferred_experience)
        
        # Extract experience from requirements
        for req in self.requirements:
            if req.requirement_type == "experience":
                experience.append(req.text)
        
        return experience
    
    def get_education_requirements(self) -> List[str]:
        """
        Get all education requirements.
        
        Returns:
            List of education requirements
        """
        education = []
        education.extend(self.education_requirements)
        
        # Extract education from requirements
        for req in self.requirements:
            if req.requirement_type == "education":
                education.append(req.text)
        
        return education
    
    def extract_years_experience(self) -> Optional[int]:
        """
        Extract minimum years of experience required.
        
        Returns:
            Minimum years of experience or None if not specified
        """
        import re
        
        all_text = self.get_all_text().lower()
        
        # Common patterns for experience requirements
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?(?:work\s*)?experience',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, all_text)
            years.extend([int(match) for match in matches])
        
        return min(years) if years else None
    
    def is_valid(self) -> bool:
        """
        Check if job description contains minimum required information.
        
        Returns:
            True if job description has essential information
        """
        has_title = bool(self.job_title.strip())
        has_description = bool(self.description and self.description.strip())
        has_raw_text = bool(self.raw_text.strip())
        has_requirements = (
            len(self.required_skills) > 0 or 
            len(self.required_experience) > 0 or
            len(self.requirements) > 0 or
            len(self.responsibilities) > 0
        )
        
        return (has_title or has_description or has_raw_text) and (has_description or has_requirements)
    
    def get_requirement_categories(self) -> Dict[str, List[JobRequirement]]:
        """
        Group requirements by category.
        
        Returns:
            Dictionary mapping categories to requirement lists
        """
        categories = {}
        for req in self.requirements:
            if req.category not in categories:
                categories[req.category] = []
            categories[req.category].append(req)
        
        return categories
    
    def get_requirement_types(self) -> Dict[str, List[JobRequirement]]:
        """
        Group requirements by type.
        
        Returns:
            Dictionary mapping types to requirement lists
        """
        types = {}
        for req in self.requirements:
            if req.requirement_type not in types:
                types[req.requirement_type] = []
            types[req.requirement_type].append(req)
        
        return types

