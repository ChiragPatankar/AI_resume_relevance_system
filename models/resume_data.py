"""
Data models for resume information.

This module contains classes for structured resume data representation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class ContactInfo:
    """Contact information extracted from resume."""
    
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "linkedin": self.linkedin,
            "github": self.github,
            "website": self.website
        }


@dataclass
class Education:
    """Education information from resume."""
    
    degree: Optional[str] = None
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    honors: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "degree": self.degree,
            "institution": self.institution,
            "field_of_study": self.field_of_study,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "gpa": self.gpa,
            "honors": self.honors,
            "description": self.description
        }
    
    def get_degree_level(self) -> str:
        """
        Determine degree level for scoring purposes.
        
        Returns:
            Standardized degree level string
        """
        if not self.degree:
            return "unknown"
        
        degree_lower = self.degree.lower()
        
        if any(term in degree_lower for term in ["phd", "ph.d", "doctorate", "doctoral"]):
            return "phd"
        elif any(term in degree_lower for term in ["master", "mba", "m.s", "m.a", "ms", "ma"]):
            return "master"
        elif any(term in degree_lower for term in ["bachelor", "b.s", "b.a", "bs", "ba"]):
            return "bachelor"
        elif any(term in degree_lower for term in ["associate", "a.s", "a.a", "as", "aa"]):
            return "associate"
        elif any(term in degree_lower for term in ["diploma", "certificate"]):
            return "diploma"
        else:
            return "certificate"


@dataclass
class Experience:
    """Work experience information from resume."""
    
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "is_current": self.is_current,
            "description": self.description,
            "responsibilities": self.responsibilities,
            "achievements": self.achievements
        }
    
    def get_duration_years(self) -> float:
        """
        Calculate experience duration in years.
        
        Returns:
            Duration in years (approximate)
        """
        try:
            # Simple estimation - can be enhanced with proper date parsing
            if self.start_date and (self.end_date or self.is_current):
                # This is a simplified calculation
                # In production, would use proper date parsing
                return 1.0  # Placeholder - implement proper date calculation
        except Exception:
            pass
        return 0.0


@dataclass 
class Skill:
    """Skill information from resume."""
    
    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years_experience: Optional[int] = None
    is_technical: bool = True
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "category": self.category,
            "proficiency": self.proficiency,
            "years_experience": self.years_experience,
            "is_technical": self.is_technical,
            "confidence_score": self.confidence_score
        }


@dataclass
class ResumeData:
    """Main class for structured resume information."""
    
    # Basic information
    contact_info: ContactInfo = field(default_factory=ContactInfo)
    
    # Professional information
    summary: Optional[str] = None
    objective: Optional[str] = None
    
    # Skills and expertise
    skills: List[Skill] = field(default_factory=list)
    
    # Experience and education
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    
    # Additional sections
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    
    # Raw and processed text
    raw_text: str = ""
    processed_text: str = ""
    
    # Metadata
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    parsing_timestamp: Optional[datetime] = None
    parsing_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.parsing_timestamp is None:
            self.parsing_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert resume data to dictionary representation.
        
        Returns:
            Dictionary containing all resume data
        """
        return {
            "contact_info": self.contact_info.to_dict(),
            "summary": self.summary,
            "objective": self.objective,
            "skills": [skill.to_dict() for skill in self.skills],
            "experience": [exp.to_dict() for exp in self.experience],
            "education": [edu.to_dict() for edu in self.education],
            "certifications": self.certifications,
            "languages": self.languages,
            "projects": self.projects,
            "publications": self.publications,
            "awards": self.awards,
            "raw_text": self.raw_text,
            "processed_text": self.processed_text,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "parsing_timestamp": self.parsing_timestamp.isoformat() if self.parsing_timestamp else None,
            "parsing_errors": self.parsing_errors
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert resume data to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResumeData':
        """
        Create ResumeData instance from dictionary.
        
        Args:
            data: Dictionary containing resume data
            
        Returns:
            ResumeData instance
        """
        contact_info = ContactInfo(**data.get("contact_info", {}))
        
        skills = [Skill(**skill_data) for skill_data in data.get("skills", [])]
        experience = [Experience(**exp_data) for exp_data in data.get("experience", [])]
        education = [Education(**edu_data) for edu_data in data.get("education", [])]
        
        parsing_timestamp = None
        if data.get("parsing_timestamp"):
            try:
                parsing_timestamp = datetime.fromisoformat(data["parsing_timestamp"])
            except ValueError:
                pass
        
        return cls(
            contact_info=contact_info,
            summary=data.get("summary"),
            objective=data.get("objective"),
            skills=skills,
            experience=experience,
            education=education,
            certifications=data.get("certifications", []),
            languages=data.get("languages", []),
            projects=data.get("projects", []),
            publications=data.get("publications", []),
            awards=data.get("awards", []),
            raw_text=data.get("raw_text", ""),
            processed_text=data.get("processed_text", ""),
            file_name=data.get("file_name"),
            file_type=data.get("file_type"),
            parsing_timestamp=parsing_timestamp,
            parsing_errors=data.get("parsing_errors", [])
        )
    
    def get_all_text(self) -> str:
        """
        Get all textual content from the resume.
        
        Returns:
            Combined text from all sections
        """
        text_parts = []
        
        if self.summary:
            text_parts.append(self.summary)
        
        if self.objective:
            text_parts.append(self.objective)
        
        # Add experience descriptions
        for exp in self.experience:
            if exp.description:
                text_parts.append(exp.description)
            text_parts.extend(exp.responsibilities)
            text_parts.extend(exp.achievements)
        
        # Add education descriptions
        for edu in self.education:
            if edu.description:
                text_parts.append(edu.description)
        
        # Add other sections
        text_parts.extend(self.certifications)
        text_parts.extend(self.publications)
        text_parts.extend(self.awards)
        
        # Add project descriptions
        for project in self.projects:
            if isinstance(project, dict) and "description" in project:
                text_parts.append(project["description"])
        
        return " ".join(text_parts)
    
    def get_skill_names(self) -> List[str]:
        """
        Get list of all skill names.
        
        Returns:
            List of skill names
        """
        return [skill.name for skill in self.skills]
    
    def get_years_experience(self) -> float:
        """
        Calculate total years of experience.
        
        Returns:
            Total years of experience
        """
        total_years = 0.0
        for exp in self.experience:
            total_years += exp.get_duration_years()
        return total_years
    
    def get_highest_education(self) -> Optional[Education]:
        """
        Get the highest level of education.
        
        Returns:
            Education object with highest degree level
        """
        if not self.education:
            return None
        
        degree_priority = {
            "phd": 6,
            "master": 5,
            "bachelor": 4,
            "associate": 3,
            "diploma": 2,
            "certificate": 1,
            "unknown": 0
        }
        
        highest_edu = None
        highest_level = -1
        
        for edu in self.education:
            level = degree_priority.get(edu.get_degree_level(), 0)
            if level > highest_level:
                highest_level = level
                highest_edu = edu
        
        return highest_edu
    
    def has_parsing_errors(self) -> bool:
        """
        Check if there were any parsing errors.
        
        Returns:
            True if parsing errors exist
        """
        return len(self.parsing_errors) > 0
    
    def is_valid(self) -> bool:
        """
        Check if resume data contains minimum required information.
        
        Returns:
            True if resume has essential information
        """
        # Basic validation - at least name or some experience/education
        has_name = self.contact_info.name is not None
        has_experience = len(self.experience) > 0
        has_education = len(self.education) > 0
        has_skills = len(self.skills) > 0
        has_text = len(self.raw_text.strip()) > 0
        
        return has_text and (has_name or has_experience or has_education or has_skills)

