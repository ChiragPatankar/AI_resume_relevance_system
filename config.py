"""
Configuration settings for the Resume Relevance Check System.

This module contains all configuration parameters and settings used throughout
the application, including model configurations, scoring weights, and system settings.

IMPORTANT: Before running the system, you need to:
1. Get a Groq API key from https://console.groq.com/
2. Replace "YOUR_GROQ_API_KEY_HERE" with your actual API key
3. Optionally, set it as an environment variable: GROQ_API_KEY
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for ML models and NLP processing."""
    
    # spaCy model configuration
    spacy_model: str = "en_core_web_sm"
    
    # Sentence transformer model for semantic similarity
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # Groq LLM configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")  # Replace with your actual Groq API key or set GROQ_API_KEY env var
    groq_model: str = "llama-3.1-70b-versatile"
    groq_temperature: float = 0.3
    groq_max_tokens: int = 1024
    
    # TF-IDF configuration
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 3)
    tfidf_stop_words: str = "english"
    
    # Text processing parameters
    min_word_length: int = 2
    max_text_length: int = 50000


@dataclass
class ScoringConfig:
    """Configuration for scoring algorithms and weights."""
    
    # Main category weights (should sum to 1.0)
    skills_weight: float = 0.4
    experience_weight: float = 0.35
    education_weight: float = 0.15
    keywords_weight: float = 0.1
    
    # Sub-scoring weights
    semantic_similarity_weight: float = 0.6
    keyword_matching_weight: float = 0.4
    
    # Minimum scores for filtering
    min_relevance_score: float = 0.0
    max_relevance_score: float = 1.0
    
    # Experience scoring parameters
    years_experience_bonus: float = 0.1  # Bonus per year of relevant experience
    max_experience_bonus: float = 0.5    # Maximum bonus from experience
    
    # Education scoring parameters
    degree_level_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.degree_level_scores is None:
            self.degree_level_scores = {
                "phd": 1.0,
                "doctorate": 1.0,
                "master": 0.8,
                "bachelor": 0.6,
                "associate": 0.4,
                "diploma": 0.3,
                "certificate": 0.2
            }


@dataclass
class ParsingConfig:
    """Configuration for resume parsing."""
    
    # Supported file formats
    supported_formats: List[str] = None
    
    # File size limits (in MB)
    max_file_size_mb: int = 10
    
    # Text extraction settings
    pdf_extraction_method: str = "pdfplumber"  # "pdfplumber" or "pypdf2"
    preserve_formatting: bool = True
    
    # Content extraction patterns
    email_pattern: str = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern: str = r'(\+\d{1,3}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}'
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".docx", ".txt", ".doc"]


@dataclass
class SkillsConfig:
    """Configuration for skill extraction and matching."""
    
    # Skill databases and sources
    use_predefined_skills: bool = True
    custom_skills_file: Optional[str] = None
    
    # Skill matching parameters
    fuzzy_match_threshold: int = 85  # Threshold for fuzzy string matching
    enable_skill_synonyms: bool = True
    enable_skill_clustering: bool = True
    
    # Common skill categories
    skill_categories: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.skill_categories is None:
            self.skill_categories = {
                "programming": [
                    "python", "java", "javascript", "c++", "c#", "php", "ruby", 
                    "go", "rust", "swift", "kotlin", "typescript", "scala"
                ],
                "web_development": [
                    "html", "css", "react", "angular", "vue", "node.js", "django", 
                    "flask", "express", "bootstrap", "jquery", "webpack"
                ],
                "data_science": [
                    "machine learning", "deep learning", "data analysis", "statistics",
                    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "sql"
                ],
                "cloud": [
                    "aws", "azure", "google cloud", "docker", "kubernetes", 
                    "terraform", "jenkins", "ci/cd"
                ],
                "databases": [
                    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                    "oracle", "sql server", "cassandra"
                ]
            }


@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""
    
    # App settings
    page_title: str = "Resume Relevance Checker"
    page_icon: str = "📄"
    layout: str = "wide"
    
    # File upload settings
    max_upload_size_mb: int = 10
    allowed_file_types: List[str] = None
    
    # Display settings
    show_debug_info: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    # Chart configurations
    chart_theme: str = "plotly"
    color_scheme: List[str] = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ["pdf", "docx", "txt"]
        
        if self.color_scheme is None:
            self.color_scheme = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
            ]


@dataclass
class SystemConfig:
    """General system configuration."""
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/resume_checker.log"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # Performance settings
    enable_multiprocessing: bool = True
    max_workers: int = 4
    
    # Cache settings
    cache_dir: str = ".cache"
    enable_model_caching: bool = True
    
    # Data directories
    data_dir: str = "data"
    models_dir: str = "data/models"
    temp_dir: str = "temp"
    
    # API settings (for future extensions)
    api_timeout: int = 30
    max_retries: int = 3


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to external configuration file
        """
        # Initialize all configuration sections
        self.model = ModelConfig()
        self.scoring = ScoringConfig()
        self.parsing = ParsingConfig()
        self.skills = SkillsConfig()
        self.ui = UIConfig()
        self.system = SystemConfig()
        
        # Load from environment variables if available
        self._load_from_env()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Create necessary directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model configuration
        if os.getenv("SPACY_MODEL"):
            self.model.spacy_model = os.getenv("SPACY_MODEL")
        
        if os.getenv("SENTENCE_TRANSFORMER_MODEL"):
            self.model.sentence_transformer_model = os.getenv("SENTENCE_TRANSFORMER_MODEL")
        
        # System configuration
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
        
        if os.getenv("MAX_WORKERS"):
            self.system.max_workers = int(os.getenv("MAX_WORKERS"))
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file."""
        # Implementation for loading from external config file
        # This can be extended to support YAML/JSON configuration files
        pass
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.system.data_dir,
            self.system.models_dir,
            self.system.temp_dir,
            self.system.cache_dir,
            os.path.dirname(self.system.log_file)
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate scoring weights sum to 1.0
        total_weight = (
            self.scoring.skills_weight + 
            self.scoring.experience_weight + 
            self.scoring.education_weight + 
            self.scoring.keywords_weight
        )
        
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            return False
        
        # Validate file size limits
        if self.parsing.max_file_size_mb <= 0:
            return False
        
        # Validate fuzzy match threshold
        if not (0 <= self.skills.fuzzy_match_threshold <= 100):
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "spacy_model": self.model.spacy_model,
            "sentence_transformer": self.model.sentence_transformer_model,
            "scoring_weights": {
                "skills": self.scoring.skills_weight,
                "experience": self.scoring.experience_weight,
                "education": self.scoring.education_weight,
                "keywords": self.scoring.keywords_weight
            },
            "supported_formats": self.parsing.supported_formats,
            "max_file_size_mb": self.parsing.max_file_size_mb,
            "log_level": self.system.log_level
        }


# Global configuration instance
config = Config()

