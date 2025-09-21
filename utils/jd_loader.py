"""
Job Description Loader Utility

This module loads and processes job descriptions from the JD folder,
making them available as sample data in the Streamlit application.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import PDF processing libraries
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)

class JDLoader:
    """Load and process job descriptions from files."""
    
    def __init__(self, jd_folder_path: str = "JD"):
        """
        Initialize JD loader.
        
        Args:
            jd_folder_path: Path to the folder containing JD files
        """
        self.jd_folder_path = Path(jd_folder_path)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using available libraries.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Try PyMuPDF first
        if FITZ_AVAILABLE:
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text += page.get_text()
                doc.close()
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")
        
        # Fall back to pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        
        # If both fail, return error message
        logger.error(f"Could not extract text from {pdf_path}")
        return f"Error: Could not extract text from {pdf_path}"
    
    def parse_jd_content(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Parse job description content and extract structured information.
        
        Args:
            text: Raw text content
            filename: Source filename
            
        Returns:
            Structured job description data
        """
        # Clean the text
        text = text.strip().replace('\n\n', '\n').replace('\t', ' ')
        
        # Try to extract job title from filename or content
        job_title = filename.replace('.pdf', '').replace('_', ' ').title()
        
        # Look for common job title patterns in text
        lines = text.split('\n')
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if line and (
                'position' in line.lower() or 
                'role' in line.lower() or 
                'engineer' in line.lower() or
                'developer' in line.lower() or
                'manager' in line.lower() or
                'analyst' in line.lower() or
                'scientist' in line.lower()
            ):
                if len(line) < 100:  # Reasonable title length
                    job_title = line
                    break
        
        # Extract company name (usually in first few lines)
        company = "Sample Company"
        for line in lines[:5]:
            line = line.strip()
            if line and (
                'company' in line.lower() or 
                'corp' in line.lower() or
                'inc' in line.lower() or
                'ltd' in line.lower() or
                'technologies' in line.lower()
            ):
                if len(line) < 100:
                    company = line
                    break
        
        # Extract skills (look for common skill keywords)
        skills = []
        skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 'sql',
            'machine learning', 'ai', 'data science', 'aws', 'azure', 'docker',
            'kubernetes', 'git', 'agile', 'scrum', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'matplotlib', 'html', 'css', 'mongodb', 'postgresql',
            'redis', 'elasticsearch', 'spark', 'hadoop', 'tableau', 'power bi'
        ]
        
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill in text_lower:
                skills.append(skill.title())
        
        return {
            'title': job_title,
            'company': company,
            'description': text,
            'required_skills': skills[:8],  # Top 8 skills
            'preferred_skills': skills[8:15] if len(skills) > 8 else [],
            'location': 'Not specified',
            'salary_range': 'Competitive',
            'source': f'JD folder - {filename}'
        }
    
    def load_sample_jds(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all job descriptions from the JD folder.
        
        Returns:
            Dictionary of job descriptions keyed by display name
        """
        sample_jds = {}
        
        if not self.jd_folder_path.exists():
            logger.warning(f"JD folder not found: {self.jd_folder_path}")
            return sample_jds
        
        # Process PDF files in JD folder
        pdf_files = list(self.jd_folder_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing JD file: {pdf_file.name}")
                
                # Extract text from PDF
                text = self.extract_text_from_pdf(str(pdf_file))
                
                if text and "Error:" not in text:
                    # Parse content
                    jd_data = self.parse_jd_content(text, pdf_file.name)
                    
                    # Create display name
                    display_name = f"{jd_data['title']} - {jd_data['company']}"
                    sample_jds[display_name] = jd_data
                    
                    logger.info(f"Successfully loaded JD: {display_name}")
                else:
                    logger.warning(f"Could not extract content from {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        return sample_jds
    
    def get_combined_sample_jds(self) -> Dict[str, Dict[str, Any]]:
        """
        Get both hardcoded and file-based sample JDs.
        
        Returns:
            Combined dictionary of all sample JDs
        """
        # Start with hardcoded samples
        hardcoded_jds = self._get_hardcoded_samples()
        
        # Add JDs from files
        file_jds = self.load_sample_jds()
        
        # Combine them
        combined_jds = {**hardcoded_jds, **file_jds}
        
        logger.info(f"Loaded {len(combined_jds)} total sample JDs")
        return combined_jds
    
    def _get_hardcoded_samples(self) -> Dict[str, Dict[str, Any]]:
        """Get hardcoded sample job descriptions."""
        return {
            "Senior Data Scientist - TechCorp": {
                "title": "Senior Data Scientist",
                "company": "TechCorp Solutions",
                "description": """We are seeking a Senior Data Scientist to join our AI/ML team.

Required Skills:
- Python programming (3+ years)
- Machine Learning (scikit-learn, TensorFlow, PyTorch)
- Data Analysis (pandas, numpy, matplotlib)
- SQL and database management
- Statistics and probability
- Deep Learning experience

Preferred Skills:
- AWS/Azure cloud platforms
- Docker and Kubernetes
- MLOps and model deployment
- Natural Language Processing
- Computer Vision
- Spark/Hadoop for big data

Experience Requirements:
- 4+ years in data science or related field
- Experience with end-to-end ML projects
- Strong problem-solving abilities
- Team collaboration experience

Education:
- Bachelor's degree in Computer Science, Statistics, Mathematics, or related field
- Master's degree preferred

Responsibilities:
- Develop and deploy machine learning models
- Analyze large datasets to extract insights
- Collaborate with cross-functional teams
- Present findings to stakeholders
- Mentor junior data scientists""",
                "required_skills": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "Data Analysis", "TensorFlow", "PyTorch"],
                "preferred_skills": ["AWS", "Docker", "Kubernetes", "NLP", "Computer Vision", "Spark"],
                "location": "San Francisco, CA",
                "salary_range": "$120,000 - $180,000"
            },
            "Full Stack Developer - InnoTech": {
                "title": "Full Stack Developer",
                "company": "InnoTech Pvt Ltd",
                "description": """Join our dynamic development team as a Full Stack Developer.

Required Skills:
- JavaScript (ES6+), HTML5, CSS3
- React.js or Angular framework
- Node.js and Express.js
- MongoDB or PostgreSQL
- RESTful API development
- Git version control

Preferred Skills:
- TypeScript
- Next.js or Vue.js
- Docker containerization
- AWS or Google Cloud
- GraphQL
- Redis caching

Experience:
- 2+ years full stack development
- Experience with responsive design
- Knowledge of agile methodology
- Team collaboration skills

Education:
- Bachelor's in Computer Science or related field
- Relevant certifications preferred

Responsibilities:
- Develop user-facing web applications
- Build and maintain server-side applications
- Collaborate with designers and product managers
- Write clean, maintainable code
- Participate in code reviews""",
                "required_skills": ["JavaScript", "React", "Node.js", "MongoDB", "REST API", "Git", "HTML", "CSS"],
                "preferred_skills": ["TypeScript", "Docker", "AWS", "GraphQL", "Redis", "Angular"],
                "location": "Austin, TX",
                "salary_range": "$85,000 - $130,000"
            }
        }

# Global instance
jd_loader = JDLoader()

