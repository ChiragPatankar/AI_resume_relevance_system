"""
Text processing module for NLP tasks and skill extraction.

This module provides the TextProcessor class for preprocessing text,
extracting skills, and performing various NLP tasks using spaCy and other libraries.
Designed for Innomatics Research Labs resume relevance checking system.
"""

import re
import logging
from typing import List, Dict, Set, Any, Optional, Tuple
from collections import Counter
import json

# NLP libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

from config import config


class TextProcessor:
    """
    Comprehensive text processor for resume and job description analysis.
    
    Provides NLP preprocessing, skill extraction, keyword identification,
    and various text analysis capabilities optimized for recruitment workflows.
    """
    
    def __init__(self):
        """Initialize the text processor with NLP models."""
        self.logger = logging.getLogger(__name__)
        self.config = config.model
        self.skills_config = config.skills
        
        # Initialize NLP models
        self.nlp = None
        self.lemmatizer = None
        self.stopwords = set()
        
        self._initialize_nlp()
        self._load_skill_databases()
        
        # Common technical abbreviations and their expansions
        self.tech_abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'db': 'database',
            'sql': 'structured query language',
            'nosql': 'not only sql',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'ci/cd': 'continuous integration continuous deployment',
            'devops': 'development operations',
            'oop': 'object oriented programming',
            'rest': 'representational state transfer',
            'json': 'javascript object notation',
            'xml': 'extensible markup language',
            'html': 'hypertext markup language',
            'css': 'cascading style sheets',
            'js': 'javascript',
            'ts': 'typescript'
        }
    
    def _initialize_nlp(self):
        """Initialize NLP libraries and models."""
        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.config.spacy_model)
                self.logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
            except OSError:
                self.logger.warning(f"spaCy model {self.config.spacy_model} not found. Using 'en_core_web_sm'")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self.logger.warning("No spaCy English model available. Using fallback processing.")
                    self.nlp = None
        else:
            self.logger.warning("spaCy not available. Using fallback text processing.")
        
        # Initialize NLTK
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                
                self.stopwords = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                self.logger.info("NLTK initialized successfully")
                
            except Exception as e:
                self.logger.warning(f"NLTK initialization failed: {str(e)}")
    
    def _load_skill_databases(self):
        """Load skill databases and create lookup structures."""
        self.all_skills = set()
        self.skill_category_map = {}
        
        # Load from config
        for category, skills in self.skills_config.skill_categories.items():
            for skill in skills:
                skill_lower = skill.lower()
                self.all_skills.add(skill_lower)
                self.skill_category_map[skill_lower] = category
        
        # Load custom skills if specified
        if self.skills_config.custom_skills_file:
            try:
                with open(self.skills_config.custom_skills_file, 'r') as f:
                    custom_skills = json.load(f)
                    for category, skills in custom_skills.items():
                        for skill in skills:
                            skill_lower = skill.lower()
                            self.all_skills.add(skill_lower)
                            self.skill_category_map[skill_lower] = category
            except Exception as e:
                self.logger.warning(f"Could not load custom skills file: {str(e)}")
        
        self.logger.info(f"Loaded {len(self.all_skills)} skills across {len(set(self.skill_category_map.values()))} categories")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        try:
            # Basic cleaning
            processed_text = text.lower()
            
            # Expand common abbreviations
            for abbr, expansion in self.tech_abbreviations.items():
                # Use word boundaries to avoid partial matches
                pattern = rf'\b{re.escape(abbr)}\b'
                processed_text = re.sub(pattern, expansion, processed_text)
            
            # Remove URLs
            processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed_text)
            
            # Remove email addresses
            processed_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', processed_text)
            
            # Remove phone numbers
            processed_text = re.sub(r'(\+\d{1,3}\s?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}', '', processed_text)
            
            # Remove special characters but keep important punctuation
            processed_text = re.sub(r'[^\w\s\+\#\.\-/]', ' ', processed_text)
            
            # Remove excessive whitespace
            processed_text = re.sub(r'\s+', ' ', processed_text)
            
            # Remove very short or very long words
            words = processed_text.split()
            filtered_words = [
                word for word in words 
                if self.config.min_word_length <= len(word) <= 50
            ]
            
            processed_text = ' '.join(filtered_words)
            
            return processed_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def extract_skills_advanced(self, text: str) -> List[Dict[str, Any]]:
        """
        Advanced skill extraction using multiple techniques.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of extracted skills with metadata
        """
        extracted_skills = []
        skill_scores = {}
        
        try:
            preprocessed_text = self.preprocess_text(text)
            
            # Method 1: Direct matching
            direct_matches = self._extract_skills_direct_match(preprocessed_text)
            for skill, score in direct_matches.items():
                skill_scores[skill] = max(skill_scores.get(skill, 0), score)
            
            # Method 2: Fuzzy matching
            if FUZZYWUZZY_AVAILABLE and self.skills_config.fuzzy_match_threshold > 0:
                fuzzy_matches = self._extract_skills_fuzzy_match(preprocessed_text)
                for skill, score in fuzzy_matches.items():
                    skill_scores[skill] = max(skill_scores.get(skill, 0), score * 0.8)  # Slightly lower confidence
            
            # Method 3: NLP-based extraction
            if self.nlp:
                nlp_matches = self._extract_skills_nlp(text)
                for skill, score in nlp_matches.items():
                    skill_scores[skill] = max(skill_scores.get(skill, 0), score * 0.9)
            
            # Convert to structured format
            for skill, score in skill_scores.items():
                extracted_skills.append({
                    'name': skill,
                    'confidence': score,
                    'category': self.skill_category_map.get(skill.lower(), 'other'),
                    'is_technical': True  # Assume all extracted skills are technical
                })
            
            # Sort by confidence
            extracted_skills.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error in advanced skill extraction: {str(e)}")
        
        return extracted_skills
    
    def _extract_skills_direct_match(self, text: str) -> Dict[str, float]:
        """Extract skills using direct string matching."""
        matches = {}
        text_lower = text.lower()
        
        for skill in self.all_skills:
            # Create patterns for different skill formats
            patterns = [
                rf'\b{re.escape(skill)}\b',  # Exact match
                rf'\b{re.escape(skill.replace(" ", ""))}\b',  # No spaces
                rf'\b{re.escape(skill.replace("-", " "))}\b',  # Dash to space
                rf'\b{re.escape(skill.replace(".", ""))}\b'   # No dots
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches[skill] = 1.0
                    break
        
        return matches
    
    def _extract_skills_fuzzy_match(self, text: str) -> Dict[str, float]:
        """Extract skills using fuzzy string matching."""
        matches = {}
        
        # Split text into potential skill phrases
        words = text.split()
        
        # Check individual words and 2-3 word phrases
        candidates = []
        for i in range(len(words)):
            candidates.append(words[i])
            if i < len(words) - 1:
                candidates.append(f"{words[i]} {words[i+1]}")
            if i < len(words) - 2:
                candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Fuzzy match against known skills
        for candidate in candidates:
            if len(candidate) < 3:  # Skip very short candidates
                continue
            
            # Find best match
            best_match = process.extractOne(
                candidate, 
                list(self.all_skills),
                score_cutoff=self.skills_config.fuzzy_match_threshold
            )
            
            if best_match:
                skill, score = best_match
                confidence = score / 100.0  # Convert to 0-1 range
                matches[skill] = confidence
        
        return matches
    
    def _extract_skills_nlp(self, text: str) -> Dict[str, float]:
        """Extract skills using NLP techniques."""
        matches = {}
        
        try:
            doc = self.nlp(text)
            
            # Extract entities that might be skills
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE']:  # Likely to be technology/skill names
                    entity_text = ent.text.lower().strip()
                    
                    # Check if entity matches known skills
                    if entity_text in self.all_skills:
                        matches[entity_text] = 0.9
                    elif FUZZYWUZZY_AVAILABLE:
                        # Fuzzy match against known skills
                        best_match = process.extractOne(
                            entity_text,
                            list(self.all_skills),
                            score_cutoff=85
                        )
                        if best_match:
                            skill, score = best_match
                            matches[skill] = (score / 100.0) * 0.8
            
            # Extract noun phrases that might be skills
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                
                # Filter out very long chunks
                if len(chunk_text.split()) <= 3:
                    if chunk_text in self.all_skills:
                        matches[chunk_text] = 0.8
        
        except Exception as e:
            self.logger.error(f"Error in NLP skill extraction: {str(e)}")
        
        return matches
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Extract important keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords with scores
        """
        keywords = []
        
        try:
            preprocessed_text = self.preprocess_text(text)
            
            if self.nlp:
                doc = self.nlp(preprocessed_text)
                
                # Extract lemmatized tokens excluding stopwords and punctuation
                tokens = [
                    token.lemma_.lower() 
                    for token in doc 
                    if not token.is_stop 
                    and not token.is_punct 
                    and not token.is_space
                    and len(token.text) > 2
                    and token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']
                ]
                
                # Count frequency
                token_counts = Counter(tokens)
                
                # Calculate TF scores
                total_tokens = len(tokens)
                for token, count in token_counts.most_common(top_k):
                    keywords.append({
                        'keyword': token,
                        'frequency': count,
                        'tf_score': count / total_tokens,
                        'is_skill': token in self.all_skills
                    })
            
            else:
                # Fallback to simple tokenization
                words = preprocessed_text.split()
                
                # Filter words
                filtered_words = [
                    word for word in words
                    if word not in self.stopwords
                    and len(word) > 2
                    and word.isalpha()
                ]
                
                word_counts = Counter(filtered_words)
                total_words = len(filtered_words)
                
                for word, count in word_counts.most_common(top_k):
                    keywords.append({
                        'keyword': word,
                        'frequency': count,
                        'tf_score': count / total_words,
                        'is_skill': word in self.all_skills
                    })
        
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
        
        return keywords
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using spaCy.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            if self.nlp and self.nlp.has_pipe('transformer'):
                # Use transformer-based similarity if available
                doc1 = self.nlp(text1[:1000])  # Limit length for performance
                doc2 = self.nlp(text2[:1000])
                return doc1.similarity(doc2)
            
            elif self.nlp:
                # Use word vector similarity
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                return doc1.similarity(doc2)
            
            else:
                # Fallback to simple overlap
                return self._calculate_word_overlap(text1, text2)
        
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap similarity as fallback."""
        words1 = set(self.preprocess_text(text1).split())
        words2 = set(self.preprocess_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """
        Extract years of experience mentioned in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Estimated years of experience or None
        """
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?(?:work\s*)?experience',
            r'(\d+)\+?\s*years?\s*in\s+',
            r'over\s+(\d+)\s*years?',
            r'more\s+than\s+(\d+)\s*years?'
        ]
        
        years = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(match) for match in matches])
        
        return max(years) if years else None
    
    def identify_job_level(self, text: str) -> str:
        """
        Identify job level based on text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Job level classification
        """
        text_lower = text.lower()
        
        # Senior level indicators
        senior_indicators = [
            'senior', 'lead', 'principal', 'architect', 'manager', 
            'director', 'head', 'chief', 'vp', 'vice president'
        ]
        
        # Junior level indicators
        junior_indicators = [
            'junior', 'entry', 'trainee', 'intern', 'graduate',
            'associate', 'assistant', 'fresher'
        ]
        
        # Mid level indicators
        mid_indicators = [
            'mid', 'intermediate', 'experienced', 'specialist'
        ]
        
        if any(indicator in text_lower for indicator in senior_indicators):
            return 'senior'
        elif any(indicator in text_lower for indicator in junior_indicators):
            return 'junior'
        elif any(indicator in text_lower for indicator in mid_indicators):
            return 'mid'
        else:
            # Try to infer from experience years
            years = self.extract_experience_years(text)
            if years:
                if years >= 7:
                    return 'senior'
                elif years >= 3:
                    return 'mid'
                else:
                    return 'junior'
            
            return 'unknown'
    
    def extract_education_level(self, text: str) -> str:
        """
        Extract highest education level from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Education level classification
        """
        text_lower = text.lower()
        
        education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'master': ['master', 'mba', 'm.s', 'm.a', 'ms', 'ma', 'msc', 'mtech', 'm.tech'],
            'bachelor': ['bachelor', 'b.s', 'b.a', 'bs', 'ba', 'bsc', 'btech', 'b.tech', 'be', 'b.e'],
            'associate': ['associate', 'a.s', 'a.a', 'as', 'aa'],
            'diploma': ['diploma', 'certificate'],
            'high_school': ['high school', 'secondary', '12th', 'grade 12']
        }
        
        for level, indicators in education_levels.items():
            if any(indicator in text_lower for indicator in indicators):
                return level
        
        return 'unknown'
    
    def calculate_keyword_density(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """
        Calculate keyword density in text.
        
        Args:
            text: Text to analyze
            keywords: List of keywords to check
            
        Returns:
            Dictionary mapping keywords to their density scores
        """
        preprocessed_text = self.preprocess_text(text)
        words = preprocessed_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {keyword: 0.0 for keyword in keywords}
        
        densities = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences
            count = 0
            
            # Check for exact matches
            pattern = rf'\b{re.escape(keyword_lower)}\b'
            matches = re.findall(pattern, preprocessed_text)
            count += len(matches)
            
            # Calculate density
            densities[keyword] = count / total_words
        
        return densities
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': 0,
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'average_word_length': 0.0,
            'readability_score': 0.0,
            'unique_words': 0,
            'vocabulary_richness': 0.0
        }
        
        try:
            words = text.split()
            
            if words:
                stats['average_word_length'] = sum(len(word) for word in words) / len(words)
                unique_words = set(word.lower() for word in words)
                stats['unique_words'] = len(unique_words)
                stats['vocabulary_richness'] = len(unique_words) / len(words)
            
            # Count sentences using spaCy if available
            if self.nlp:
                doc = self.nlp(text)
                stats['sentence_count'] = len(list(doc.sents))
            else:
                # Simple sentence counting
                stats['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
            
            # Basic readability score (simplified Flesch formula)
            if stats['sentence_count'] > 0 and stats['word_count'] > 0:
                avg_sentence_length = stats['word_count'] / stats['sentence_count']
                avg_syllables = stats['average_word_length'] * 1.5  # Rough estimate
                stats['readability_score'] = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        except Exception as e:
            self.logger.error(f"Error calculating text statistics: {str(e)}")
        
        return stats
