"""
Main resume parser for the Resume Relevance Check System.

This module provides the ResumeParser class that coordinates text extraction,
content parsing, and structured data creation from resume files. Designed
specifically for Innomatics Research Labs requirements.
"""

import os
import re
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

from models.resume_data import ResumeData, ContactInfo, Education, Experience, Skill
from .text_extractor import TextExtractor
from config import config


class ResumeParser:
    """
    Main resume parser class for extracting structured data from resumes.
    
    Handles parsing of PDF, DOCX, and TXT resume formats and extracts
    structured information including contact details, skills, experience,
    and education for the Innomatics placement system.
    """
    
    def __init__(self):
        """Initialize the resume parser."""
        self.logger = logging.getLogger(__name__)
        self.text_extractor = TextExtractor()
        self.config = config.parsing
        
        # Initialize skill categories for better extraction
        self.skill_categories = config.skills.skill_categories
        
        # Common section headers to identify resume sections
        self.section_patterns = {
            'experience': [
                r'experience', r'work\s+experience', r'professional\s+experience',
                r'employment', r'career', r'work\s+history', r'professional\s+background',
                r'employment\s+history', r'job\s+history', r'positions', r'roles'
            ],
            'education': [
                r'education', r'educational\s+background', r'academic\s+background', 
                r'qualifications', r'academic\s+qualifications', r'degrees',
                r'university', r'college', r'school', r'academic', r'learning'
            ],
            'skills': [
                r'skills', r'technical\s+skills', r'core\s+competencies', r'competencies',
                r'expertise', r'technologies', r'proficiencies', r'abilities',
                r'capabilities', r'knowledge', r'programming', r'languages'
            ],
            'projects': [
                r'projects', r'key\s+projects', r'project\s+experience', r'portfolio',
                r'notable\s+projects', r'academic\s+projects', r'personal\s+projects'
            ],
            'certifications': [
                r'certifications', r'certificates', r'professional\s+certifications',
                r'licenses', r'credentials', r'awards', r'achievements'
            ]
        }
    
    def parse_resume(self, file_path: str) -> Optional[ResumeData]:
        """
        Parse a resume file and extract structured data.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            ResumeData object with extracted information or None if parsing fails
        """
        try:
            start_time = datetime.now()
            
            # Validate file path
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None
            
            file_path_obj = Path(file_path)
            
            # Check file format support
            file_extension = file_path_obj.suffix.lower()
            if file_extension not in self.config.supported_formats:
                self.logger.error(f"Unsupported file format: {file_extension}")
                return None
            
            # Extract text from file
            extraction_result = self.text_extractor.extract_text(file_path)
            
            if not extraction_result["success"]:
                self.logger.error(f"Text extraction failed for {file_path}: {extraction_result['errors']}")
                return None
            
            raw_text = extraction_result["text"]
            
            # Validate extraction
            if not self.text_extractor.validate_extraction(extraction_result):
                self.logger.warning(f"Extracted text validation failed for {file_path}")
                return None
            
            # Create ResumeData object
            resume_data = ResumeData(
                raw_text=raw_text,
                file_name=file_path_obj.name,
                file_type=file_extension,
                parsing_timestamp=datetime.now()
            )
            
            # Extract structured information
            try:
                # Extract contact information
                resume_data.contact_info = self._extract_contact_info(raw_text)
                
                # Debug logging for name extraction
                self.logger.info(f"Extracted name: '{resume_data.contact_info.name}' from {file_path}")
                
                # Fallback: try to extract name from email or filename if no name found
                if not resume_data.contact_info.name or resume_data.contact_info.name.strip() == "":
                    # First try email-based name extraction
                    if resume_data.contact_info.email:
                        email_name = self._extract_name_from_email_simple(resume_data.contact_info.email)
                        if email_name:
                            resume_data.contact_info.name = email_name
                            self.logger.info(f"Used email-based name: '{email_name}'")
                    
                    # If still no name, try filename
                    if not resume_data.contact_info.name:
                        filename_name = self._extract_name_from_filename(file_path)
                        if filename_name:
                            resume_data.contact_info.name = filename_name
                            self.logger.info(f"Used filename-based name: '{filename_name}'")
                
                # Extract sections
                sections = self._identify_sections(raw_text)
                self.logger.info(f"Identified sections: {list(sections.keys())}")
                
                # Log section content lengths for debugging
                for section_name, content in sections.items():
                    self.logger.info(f"Section '{section_name}': {len(content)} characters")
                
                # Extract skills
                resume_data.skills = self._extract_skills(raw_text, sections.get('skills', ''))
                
                # Extract experience (with fallback)
                experience_text = sections.get('experience', '')
                if not experience_text:
                    self.logger.info("Experience section not found, using fallback extraction")
                    experience_text = self._extract_section_fallback(raw_text, 'experience')
                resume_data.experience = self._extract_experience(experience_text)
                self.logger.info(f"Extracted {len(resume_data.experience)} experience entries")
                
                # Extract education (with fallback)
                education_text = sections.get('education', '')
                if not education_text:
                    self.logger.info("Education section not found, using fallback extraction")
                    education_text = self._extract_section_fallback(raw_text, 'education')
                resume_data.education = self._extract_education(education_text)
                self.logger.info(f"Extracted {len(resume_data.education)} education entries")
                
                # Extract other sections
                resume_data.certifications = self._extract_certifications(
                    sections.get('certifications', '')
                )
                resume_data.projects = self._extract_projects(sections.get('projects', ''))
                
                # Extract summary/objective
                resume_data.summary, resume_data.objective = self._extract_summary_objective(raw_text)
                
                # Set processed text (cleaned version)
                resume_data.processed_text = self._clean_and_normalize_text(raw_text)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Successfully parsed {file_path} in {processing_time:.2f} seconds")
                
                return resume_data
                
            except Exception as e:
                self.logger.error(f"Error during structured extraction for {file_path}: {str(e)}")
                resume_data.parsing_errors.append(f"Structured extraction error: {str(e)}")
                return resume_data  # Return partial data
                
        except Exception as e:
            self.logger.error(f"Critical error parsing {file_path}: {str(e)}")
            return None
    
    def _extract_contact_info(self, text: str) -> ContactInfo:
        """Extract contact information from resume text."""
        contact_info = ContactInfo()
        
        try:
            # Extract basic contact info using text extractor
            extracted_contact = self.text_extractor.extract_contact_info(text)
            contact_info.email = extracted_contact.get("email")
            contact_info.phone = extracted_contact.get("phone")
            contact_info.linkedin = extracted_contact.get("linkedin")
            contact_info.github = extracted_contact.get("github")
            
            # Extract name (usually at the beginning)
            name_match = self._extract_name(text)
            if name_match:
                contact_info.name = name_match
            
            # Extract address (basic pattern)
            address_patterns = [
                r'(?:Address|Location):\s*([^\n]+)',
                r'\b\d+[^,\n]+,\s*[^,\n]+,\s*[^,\n]+\b'
            ]
            
            for pattern in address_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    contact_info.address = match.group(1) if pattern.startswith('(?:') else match.group()
                    break
            
        except Exception as e:
            self.logger.error(f"Error extracting contact info: {str(e)}")
        
        return contact_info
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name from resume text with enhanced logic."""
        lines = text.strip().split('\n')
        
        # First, try to find name in first 10 lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            # Skip obvious non-name lines
            skip_keywords = [
                'resume', 'cv', 'curriculum', 'profile', 'summary', 'objective',
                '@', 'phone', 'email', 'address', 'linkedin', 'github',
                'experience', 'education', 'skills', 'projects', 'contact',
                'tel:', 'mob:', 'mobile:', 'www.', 'http', '.com', '.in'
            ]
            
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Skip lines with mostly numbers or special characters
            if len([c for c in line if c.isdigit() or c in '+-()[]{}|']) > len(line) * 0.3:
                continue
            
            # Enhanced name detection
            words = line.split()
            
            # Filter out non-name words
            name_words = []
            for word in words:
                # Clean word of punctuation
                clean_word = ''.join(c for c in word if c.isalpha())
                if clean_word and len(clean_word) > 1:
                    name_words.append(clean_word)
            
            # Check if this looks like a name
            if self._looks_like_name(name_words, line):
                return ' '.join(name_words)
        
        # Fallback: try to extract from email if available
        return self._extract_name_from_email(text)
    
    def _looks_like_name(self, words: list, original_line: str) -> bool:
        """Check if a list of words looks like a person's name."""
        if not words or len(words) < 1 or len(words) > 5:
            return False
        
        # Each word should start with uppercase and be reasonable length
        for word in words:
            if not word[0].isupper() or len(word) < 2 or len(word) > 20:
                return False
        
        # Total length should be reasonable for a name
        total_length = len(' '.join(words))
        if total_length < 3 or total_length > 50:
            return False
        
        # Common name patterns
        if len(words) == 1:
            # Single word names are possible but should be reasonably long
            return len(words[0]) >= 3 and words[0].isalpha()
        elif len(words) == 2:
            # First Name Last Name - most common
            return all(len(word) >= 2 for word in words)
        elif len(words) == 3:
            # First Middle Last or First Last Suffix
            return all(len(word) >= 2 for word in words)
        else:
            # 4+ words can be names but less common
            return all(len(word) >= 2 for word in words)
    
    def _extract_name_from_email(self, text: str) -> Optional[str]:
        """Try to extract name from email address as fallback."""
        import re
        
        # Find email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        for email in emails:
            # Extract name part before @
            username = email.split('@')[0]
            
            # Common patterns: firstname.lastname, firstnamelastname, firstname_lastname
            if '.' in username:
                parts = username.split('.')
            elif '_' in username:
                parts = username.split('_')
            else:
                # Try to split camelCase or find name patterns
                parts = [username]
            
            # Clean and capitalize parts
            name_parts = []
            for part in parts:
                if part.isalpha() and len(part) > 1:
                    name_parts.append(part.capitalize())
            
            if len(name_parts) >= 2:
                return ' '.join(name_parts)
        
        return None
    
    def _extract_name_from_email_simple(self, email: str) -> Optional[str]:
        """Extract name from a single email address."""
        if not email or '@' not in email:
            return None
            
        # Extract username part before @
        username = email.split('@')[0].lower()
        
        # Common patterns to split the username
        name_parts = []
        
        if '.' in username:
            # firstname.lastname pattern
            parts = username.split('.')
            for part in parts:
                if part.isalpha() and len(part) > 1:
                    name_parts.append(part.capitalize())
        elif any(char.isdigit() for char in username):
            # Remove numbers and split
            clean_username = ''.join(c for c in username if c.isalpha())
            if len(clean_username) > 2:
                # Try to identify common name patterns
                if len(clean_username) <= 15:  # Reasonable length for concatenated names
                    # For now, just capitalize the whole thing
                    name_parts.append(clean_username.capitalize())
        elif len(username) > 1 and username.isalpha():
            # Single word username
            name_parts.append(username.capitalize())
        
        # Try to split camelCase or common patterns
        if len(name_parts) == 1 and len(name_parts[0]) > 6:
            name = name_parts[0]
            # Look for capital letters in the middle (camelCase)
            split_points = []
            for i, char in enumerate(name[1:], 1):
                if char.isupper():
                    split_points.append(i)
            
            if split_points:
                parts = []
                start = 0
                for point in split_points:
                    parts.append(name[start:point])
                    start = point
                parts.append(name[start:])
                name_parts = [part for part in parts if len(part) > 1]
        
        if len(name_parts) >= 1:
            return ' '.join(name_parts[:3])  # Max 3 parts
            
        return None
    
    def _extract_name_from_filename(self, file_path: str) -> Optional[str]:
        """Extract name from filename as a fallback method."""
        import os
        
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Clean filename
        # Remove common resume keywords
        clean_name = filename.lower()
        for keyword in ['resume', 'cv', 'curriculum']:
            clean_name = clean_name.replace(keyword, '')
        
        # Clean separators and numbers
        clean_name = clean_name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        clean_name = ''.join(c if c.isalpha() or c.isspace() else ' ' for c in clean_name)
        
        # Split into words and capitalize
        words = [word.strip().capitalize() for word in clean_name.split() if word.strip() and len(word.strip()) > 1]
        
        # Filter out numbers and single characters
        name_words = [word for word in words if word.isalpha() and len(word) > 1]
        
        if len(name_words) >= 1:
            return ' '.join(name_words[:4])  # Take first 4 words max
        
        return None
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract different sections of the resume."""
        sections = {}
        
        # Split text into lines for section identification
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                if current_section:
                    section_content.append('')
                continue
            
            # Check if line is a section header
            detected_section = self._detect_section_header(line_clean)
            
            if detected_section:
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                # Start new section
                current_section = detected_section
                section_content = []
            else:
                # Add content to current section
                if current_section:
                    section_content.append(line_clean)
        
        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content).strip()
        
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header."""
        line_lower = line.lower()
        
        # Remove common formatting characters and clean up
        line_clean = re.sub(r'[:\-_=*#\(\)\[\]]+', '', line_lower).strip()
        
        # Try exact matches first
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                # Exact pattern match
                if re.search(f'^{pattern}$', line_clean, re.IGNORECASE):
                    return section
                
                # Partial matches for short lines
                if len(line_clean) <= 25 and pattern.replace(r'\s+', ' ') in line_clean:
                    return section
                
                # Contains match for very short headers
                if len(line_clean) <= 15:
                    pattern_simple = pattern.replace(r'\s+', ' ').replace(r'(?:', '').replace(r')?', '')
                    if pattern_simple in line_clean:
                        return section
        
        # Fallback: check for common single words
        fallback_keywords = {
            'experience': ['experience', 'work', 'employment', 'career', 'jobs'],
            'education': ['education', 'academic', 'university', 'college', 'degree'],
            'skills': ['skills', 'technical', 'programming', 'languages', 'technologies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'awards']
        }
        
        for section, keywords in fallback_keywords.items():
            if any(keyword == line_clean for keyword in keywords):
                return section
        
        return None
    
    def _extract_section_fallback(self, text: str, section_type: str) -> str:
        """
        Fallback method to extract section content when header detection fails.
        Uses keyword-based text extraction.
        """
        if section_type == 'experience':
            # Look for experience-related content
            experience_keywords = [
                'company', 'position', 'role', 'responsibilities', 'achievements',
                'worked', 'managed', 'developed', 'led', 'years', 'month',
                'present', 'current', '20\\d{2}', '19\\d{2}'  # Years
            ]
            
            # Find lines with experience indicators
            lines = text.split('\n')
            experience_lines = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(re.search(keyword, line_lower) for keyword in experience_keywords):
                    # Include surrounding context
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    experience_lines.extend(lines[start:end])
            
            return '\n'.join(set(experience_lines))  # Remove duplicates
            
        elif section_type == 'education':
            # Look for education-related content
            education_keywords = [
                'university', 'college', 'school', 'degree', 'bachelor', 'master',
                'phd', 'doctorate', 'diploma', 'certificate', 'graduation',
                'graduated', 'gpa', 'cgpa', 'b\\.?tech', 'm\\.?tech', 'mba'
            ]
            
            lines = text.split('\n')
            education_lines = []
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(re.search(keyword, line_lower) for keyword in education_keywords):
                    # Include surrounding context
                    start = max(0, i-1)
                    end = min(len(lines), i+2)
                    education_lines.extend(lines[start:end])
            
            return '\n'.join(set(education_lines))  # Remove duplicates
        
        return ""
    
    def _extract_skills(self, full_text: str, skills_section: str) -> List[Skill]:
        """Extract skills from resume text."""
        skills = []
        skill_names = set()  # To avoid duplicates
        
        try:
            # Combine skills section and full text for comprehensive extraction
            text_to_analyze = f"{skills_section}\n{full_text}"
            
            # Extract from predefined skill categories
            for category, category_skills in self.skill_categories.items():
                for skill_name in category_skills:
                    # Look for exact matches and variations
                    patterns = [
                        rf'\b{re.escape(skill_name)}\b',
                        rf'\b{re.escape(skill_name.replace(" ", ""))}\b',  # No spaces
                        rf'\b{re.escape(skill_name.replace("-", " "))}\b'   # Dash to space
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, text_to_analyze, re.IGNORECASE):
                            if skill_name.lower() not in skill_names:
                                skills.append(Skill(
                                    name=skill_name,
                                    category=category,
                                    is_technical=True,
                                    confidence_score=0.9
                                ))
                                skill_names.add(skill_name.lower())
                            break
            
            # Extract additional skills from skills section using patterns
            if skills_section:
                additional_skills = self._extract_skills_from_section(skills_section)
                for skill in additional_skills:
                    if skill.name.lower() not in skill_names:
                        skills.append(skill)
                        skill_names.add(skill.name.lower())
            
        except Exception as e:
            self.logger.error(f"Error extracting skills: {str(e)}")
        
        return skills
    
    def _extract_skills_from_section(self, skills_section: str) -> List[Skill]:
        """Extract skills from dedicated skills section."""
        skills = []
        
        # Common skill separators
        separators = [',', ';', '|', '•', '◦', '-', '\n']
        
        # Replace separators with commas for uniform processing
        processed_text = skills_section
        for separator in separators:
            processed_text = processed_text.replace(separator, ',')
        
        # Extract individual skills
        potential_skills = [s.strip() for s in processed_text.split(',')]
        
        for skill_text in potential_skills:
            if skill_text and len(skill_text) > 1 and len(skill_text) < 50:
                # Clean skill name
                skill_name = re.sub(r'^\W+|\W+$', '', skill_text)
                
                if skill_name and len(skill_name) > 1:
                    skills.append(Skill(
                        name=skill_name,
                        is_technical=True,
                        confidence_score=0.7
                    ))
        
        return skills
    
    def _extract_experience(self, experience_section: str) -> List[Experience]:
        """Extract work experience from resume text."""
        experiences = []
        
        if not experience_section:
            return experiences
        
        try:
            # Split by potential job entries (look for company/position patterns)
            job_blocks = self._split_experience_entries(experience_section)
            
            for block in job_blocks:
                experience = self._parse_experience_block(block)
                if experience:
                    experiences.append(experience)
                    
        except Exception as e:
            self.logger.error(f"Error extracting experience: {str(e)}")
        
        return experiences
    
    def _split_experience_entries(self, text: str) -> List[str]:
        """Split experience section into individual job entries."""
        if not text.strip():
            return []
        
        # Look for patterns that indicate new job entries
        job_separators = [
            r'\n(?=[A-Z][^,\n]+(?:,|\s+\||\s+at\s+|\s+-\s+)[A-Z][^,\n]+)',  # Position, Company pattern
            r'\n(?=\d{4}\s*-\s*(?:\d{4}|present|current))',  # Date range pattern  
            r'\n(?=[A-Z][^,\n]+\n[A-Z][^,\n]+)',  # Position\nCompany pattern
            r'\n(?=\w+\s+\d{4}\s*-\s*(?:\w+\s+\d{4}|present|current))',  # Month Year - Month Year
            r'\n\n+',  # Double line breaks often separate entries
        ]
        
        blocks = [text]
        
        for pattern in job_separators:
            new_blocks = []
            for block in blocks:
                if block.strip():
                    split_blocks = re.split(pattern, block, flags=re.IGNORECASE)
                    new_blocks.extend([b.strip() for b in split_blocks if b.strip()])
            blocks = new_blocks
        
        # Filter out very short blocks (likely not complete job entries)
        filtered_blocks = [block for block in blocks if len(block.split()) > 3]
        
        return filtered_blocks if filtered_blocks else [text]  # Return original if filtering fails
    
    def _parse_experience_block(self, block: str) -> Optional[Experience]:
        """Parse a single experience block."""
        if not block.strip():
            return None
        
        experience = Experience()
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        try:
            # First line often contains position and/or company
            first_line = lines[0]
            
            # Look for position | company or position at company patterns
            if ' | ' in first_line:
                parts = first_line.split(' | ')
                experience.job_title = parts[0].strip()
                if len(parts) > 1:
                    experience.company = parts[1].strip()
            elif ' at ' in first_line:
                parts = first_line.split(' at ')
                experience.job_title = parts[0].strip()
                if len(parts) > 1:
                    experience.company = parts[1].strip()
            else:
                experience.job_title = first_line
            
            # Look for dates in the first few lines
            for line in lines[:3]:
                dates = self._extract_dates(line)
                if dates:
                    experience.start_date = dates.get('start')
                    experience.end_date = dates.get('end')
                    experience.is_current = dates.get('is_current', False)
                    break
            
            # Extract responsibilities and achievements
            description_lines = []
            for line in lines[1:]:
                # Skip lines that look like dates or location
                if self._extract_dates(line) or self._looks_like_location(line):
                    continue
                description_lines.append(line)
            
            if description_lines:
                experience.description = '\n'.join(description_lines)
                
                # Split into responsibilities and achievements
                responsibilities, achievements = self._categorize_experience_items(description_lines)
                experience.responsibilities = responsibilities
                experience.achievements = achievements
            
        except Exception as e:
            self.logger.error(f"Error parsing experience block: {str(e)}")
        
        return experience if experience.job_title else None
    
    def _extract_dates(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract date ranges from text."""
        # Common date patterns
        patterns = [
            r'(\d{4})\s*-\s*(\d{4})',
            r'(\d{4})\s*-\s*(present|current)',
            r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4})',
            r'(\w+\s+\d{4})\s*-\s*(present|current)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_date = match.group(1)
                end_date = match.group(2)
                is_current = end_date.lower() in ['present', 'current']
                
                return {
                    'start': start_date,
                    'end': end_date if not is_current else None,
                    'is_current': is_current
                }
        
        return None
    
    def _looks_like_location(self, text: str) -> bool:
        """Check if text looks like a location."""
        location_indicators = ['city', 'state', 'country', 'remote', 'onsite']
        text_lower = text.lower()
        
        # Check for common location patterns
        if re.search(r'\b\w+,\s*\w+\b', text):  # City, State pattern
            return True
        
        if any(indicator in text_lower for indicator in location_indicators):
            return True
        
        return False
    
    def _categorize_experience_items(self, lines: List[str]) -> tuple[List[str], List[str]]:
        """Categorize experience items into responsibilities and achievements."""
        responsibilities = []
        achievements = []
        
        achievement_indicators = [
            'achieved', 'improved', 'increased', 'reduced', 'saved', 'generated',
            'led', 'managed', 'delivered', 'implemented', 'created', 'developed',
            'optimized', 'streamlined', 'enhanced', 'awarded', 'recognized'
        ]
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line indicates an achievement
            is_achievement = any(indicator in line_lower for indicator in achievement_indicators)
            
            # Also check for numbers/percentages which often indicate achievements
            has_metrics = re.search(r'\d+%|\$\d+|\d+x|by \d+', line)
            
            if is_achievement or has_metrics:
                achievements.append(line)
            else:
                responsibilities.append(line)
        
        return responsibilities, achievements
    
    def _extract_education(self, education_section: str) -> List[Education]:
        """Extract education information from resume text."""
        educations = []
        
        if not education_section:
            return educations
        
        try:
            # Split by potential education entries
            edu_blocks = self._split_education_entries(education_section)
            
            for block in edu_blocks:
                education = self._parse_education_block(block)
                if education:
                    educations.append(education)
                    
        except Exception as e:
            self.logger.error(f"Error extracting education: {str(e)}")
        
        return educations
    
    def _split_education_entries(self, text: str) -> List[str]:
        """Split education section into individual entries."""
        # Look for patterns that indicate new education entries
        edu_separators = [
            r'\n(?=(?:Bachelor|Master|PhD|B\.S|B\.A|M\.S|M\.A|MBA))',
            r'\n(?=\d{4}\s*-\s*\d{4})',  # Date pattern
            r'\n(?=[A-Z][^,\n]+University|[A-Z][^,\n]+College|[A-Z][^,\n]+Institute)',
        ]
        
        blocks = [text]
        
        for pattern in edu_separators:
            new_blocks = []
            for block in blocks:
                split_blocks = re.split(pattern, block, flags=re.IGNORECASE)
                new_blocks.extend([b.strip() for b in split_blocks if b.strip()])
            blocks = new_blocks
        
        return blocks
    
    def _parse_education_block(self, block: str) -> Optional[Education]:
        """Parse a single education block."""
        if not block.strip():
            return None
        
        education = Education()
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        try:
            # Look for degree, institution, and dates
            for line in lines:
                # Check for degree patterns
                degree_patterns = [
                    r'(Bachelor|Master|PhD|B\.S|B\.A|M\.S|M\.A|MBA).*?(?:in\s+)?([^,\n]+)',
                    r'(Diploma|Certificate).*?(?:in\s+)?([^,\n]+)'
                ]
                
                for pattern in degree_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        education.degree = match.group(1)
                        if len(match.groups()) > 1:
                            education.field_of_study = match.group(2).strip()
                        break
                
                # Check for institution
                institution_patterns = [
                    r'(University|College|Institute|School)\s+of\s+([^,\n]+)',
                    r'([^,\n]+\s+(?:University|College|Institute|School))',
                ]
                
                for pattern in institution_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        education.institution = match.group().strip()
                        break
                
                # Check for dates
                dates = self._extract_dates(line)
                if dates:
                    education.start_date = dates.get('start')
                    education.end_date = dates.get('end')
                
                # Check for GPA
                gpa_match = re.search(r'GPA:?\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if gpa_match:
                    education.gpa = gpa_match.group(1)
            
        except Exception as e:
            self.logger.error(f"Error parsing education block: {str(e)}")
        
        return education if education.degree or education.institution else None
    
    def _extract_certifications(self, certifications_section: str) -> List[str]:
        """Extract certifications from resume text."""
        certifications = []
        
        if not certifications_section:
            return certifications
        
        # Split by common separators
        lines = certifications_section.replace(',', '\n').replace(';', '\n').split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:
                # Clean up common prefixes/suffixes
                line = re.sub(r'^\W+|\W+$', '', line)
                if line:
                    certifications.append(line)
        
        return certifications
    
    def _extract_projects(self, projects_section: str) -> List[Dict[str, Any]]:
        """Extract projects from resume text."""
        projects = []
        
        if not projects_section:
            return projects
        
        try:
            # Split by potential project entries
            project_blocks = re.split(r'\n(?=[A-Z][^:\n]+:)', projects_section)
            
            for block in project_blocks:
                if not block.strip():
                    continue
                
                lines = [line.strip() for line in block.split('\n') if line.strip()]
                if not lines:
                    continue
                
                project = {}
                
                # First line is likely the project title
                if ':' in lines[0]:
                    title, description = lines[0].split(':', 1)
                    project['title'] = title.strip()
                    project['description'] = description.strip()
                else:
                    project['title'] = lines[0]
                    project['description'] = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                
                if project['title']:
                    projects.append(project)
                    
        except Exception as e:
            self.logger.error(f"Error extracting projects: {str(e)}")
        
        return projects
    
    def _extract_summary_objective(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract summary or objective from resume text."""
        summary = None
        objective = None
        
        # Look for summary/objective sections in first part of resume
        first_part = '\n'.join(text.split('\n')[:20])  # First 20 lines
        
        # Summary patterns
        summary_patterns = [
            r'(?:Professional\s+)?Summary[:\-\s]*\n?([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z][^\n]*:|$)',
            r'Profile[:\-\s]*\n?([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z][^\n]*:|$)',
            r'About[:\-\s]*\n?([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z][^\n]*:|$)'
        ]
        
        # Objective patterns
        objective_patterns = [
            r'(?:Career\s+)?Objective[:\-\s]*\n?([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z][^\n]*:|$)',
            r'Goal[:\-\s]*\n?([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z][^\n]*:|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, first_part, re.IGNORECASE | re.MULTILINE)
            if match:
                summary = match.group(1).strip()
                break
        
        for pattern in objective_patterns:
            match = re.search(pattern, first_part, re.IGNORECASE | re.MULTILINE)
            if match:
                objective = match.group(1).strip()
                break
        
        return summary, objective
    
    def _clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def validate_resume_data(self, resume_data: ResumeData) -> bool:
        """
        Validate that resume data contains minimum required information.
        
        Args:
            resume_data: ResumeData object to validate
            
        Returns:
            True if resume data is valid for processing
        """
        if not resume_data:
            return False
        
        # Check for basic content
        if not resume_data.raw_text or len(resume_data.raw_text.strip()) < 100:
            return False
        
        # Check for at least some structured data
        has_contact = resume_data.contact_info.name or resume_data.contact_info.email
        has_experience = len(resume_data.experience) > 0
        has_education = len(resume_data.education) > 0
        has_skills = len(resume_data.skills) > 0
        
        # Should have at least 2 of the above categories
        valid_sections = sum([has_contact, has_experience, has_education, has_skills])
        
        return valid_sections >= 2

