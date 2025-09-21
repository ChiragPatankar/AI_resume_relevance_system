"""
Groq LLM enhancement module for advanced resume analysis.

This module provides optional LLM-powered enhancements using Groq's fast
inference API for more sophisticated resume-job matching analysis.
"""

import logging
import json
from typing import Dict, List, Optional, Any
import os
from dataclasses import asdict

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from models.resume_data import ResumeData
from models.job_description import JobDescription
from models.scoring_result import ScoringResult


class GroqEnhancer:
    """
    Enhanced resume analysis using Groq's fast LLM inference.
    
    Provides optional LLM-powered insights while maintaining compatibility
    with the base scoring system.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq enhancer.
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        if not GROQ_AVAILABLE:
            self.logger.warning("Groq library not available. Install with: pip install groq")
            return
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not api_key:
            self.logger.warning("No Groq API key provided. LLM enhancements disabled.")
            return
        
        try:
            self.client = Groq(api_key=api_key)
            self.logger.info("Groq client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Groq enhancement is available."""
        return self.client is not None
    
    def enhance_scoring_explanation(
        self, 
        base_result: ScoringResult,
        resume_data: ResumeData,
        job_description: JobDescription
    ) -> ScoringResult:
        """
        Enhance scoring explanation using Groq LLM.
        
        Args:
            base_result: Base scoring result from hybrid scorer
            resume_data: Resume data
            job_description: Job description
            
        Returns:
            Enhanced scoring result with improved explanations
        """
        if not self.is_available():
            self.logger.warning("Groq not available, returning base result")
            return base_result
        
        try:
            # Create enhanced explanation using LLM
            enhanced_explanation = self._generate_enhanced_explanation(
                base_result, resume_data, job_description
            )
            
            # Update the explanation in the result
            if enhanced_explanation:
                base_result.explanation.summary = enhanced_explanation.get("summary", base_result.explanation.summary)
                base_result.explanation.strengths = enhanced_explanation.get("strengths", base_result.explanation.strengths)
                base_result.explanation.weaknesses = enhanced_explanation.get("weaknesses", base_result.explanation.weaknesses)
                base_result.explanation.recommendations = enhanced_explanation.get("recommendations", base_result.explanation.recommendations)
                base_result.explanation.improvement_suggestions = enhanced_explanation.get("improvement_suggestions", base_result.explanation.improvement_suggestions)
            
            return base_result
            
        except Exception as e:
            self.logger.error(f"Error enhancing with Groq: {str(e)}")
            return base_result
    
    def _generate_enhanced_explanation(
        self,
        base_result: ScoringResult,
        resume_data: ResumeData,
        job_description: JobDescription
    ) -> Optional[Dict[str, Any]]:
        """Generate enhanced explanation using Groq LLM."""
        
        # Prepare context for LLM
        context = self._prepare_context(base_result, resume_data, job_description)
        
        prompt = f"""You are an expert HR analyst for Innomatics Research Labs, evaluating resume-job fit.

CONTEXT:
Job Title: {job_description.job_title}
Company: {job_description.company}
Overall Score: {base_result.overall_score:.2f} ({base_result.get_score_category()})

CANDIDATE PROFILE:
{context['candidate_summary']}

JOB REQUIREMENTS:
{context['job_summary']}

CURRENT ANALYSIS:
- Skills Score: {base_result.breakdown.skills_score:.2f}
- Experience Score: {base_result.breakdown.experience_score:.2f}
- Education Score: {base_result.breakdown.education_score:.2f}
- Matched Skills: {', '.join(base_result.breakdown.matched_skills[:5])}
- Missing Skills: {', '.join(base_result.breakdown.missing_skills[:5])}

TASK:
Provide a professional analysis in JSON format with these fields:
1. "summary": Brief overall assessment (2-3 sentences)
2. "strengths": List of 3-4 key candidate strengths
3. "weaknesses": List of 3-4 areas for improvement
4. "recommendations": List of 3-4 actionable recommendations for the candidate
5. "improvement_suggestions": List of 3-4 specific learning/development suggestions

Focus on practical insights for both recruiters and candidates. Be constructive and specific.

Response format: {{"summary": "...", "strengths": [...], "weaknesses": [...], "recommendations": [...], "improvement_suggestions": [...]}}"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",  # Fast Groq model
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst providing structured resume analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                enhanced_explanation = json.loads(response_text)
                return enhanced_explanation
            except json.JSONDecodeError:
                # Fallback: extract insights from text response
                return self._parse_text_response(response_text)
                
        except Exception as e:
            self.logger.error(f"Groq API call failed: {str(e)}")
            return None
    
    def _prepare_context(
        self,
        base_result: ScoringResult,
        resume_data: ResumeData,
        job_description: JobDescription
    ) -> Dict[str, str]:
        """Prepare context information for LLM analysis."""
        
        # Candidate summary
        candidate_summary = f"""
Name: {resume_data.contact_info.name or 'Not provided'}
Skills: {', '.join(resume_data.get_skill_names()[:10])}
Experience: {len(resume_data.experience)} positions
Education: {resume_data.get_highest_education().degree if resume_data.get_highest_education() else 'Not specified'}
        """.strip()
        
        # Job summary
        job_summary = f"""
Required Skills: {', '.join(job_description.required_skills[:10])}
Preferred Skills: {', '.join(job_description.preferred_skills[:5])}
Experience Level: {job_description.extract_years_experience() or 'Not specified'} years
Education: {', '.join(job_description.education_requirements[:3])}
        """.strip()
        
        return {
            "candidate_summary": candidate_summary,
            "job_summary": job_summary
        }
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response as fallback when JSON parsing fails."""
        
        # Simple text parsing fallback
        lines = response_text.split('\n')
        
        result = {
            "summary": "Enhanced analysis provided by AI assistant.",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "improvement_suggestions": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "strength" in line.lower():
                current_section = "strengths"
            elif "weakness" in line.lower() or "area" in line.lower():
                current_section = "weaknesses"
            elif "recommendation" in line.lower():
                current_section = "recommendations"
            elif "suggestion" in line.lower() or "improvement" in line.lower():
                current_section = "improvement_suggestions"
            elif line.startswith('-') or line.startswith('•'):
                # List item
                if current_section and current_section in result:
                    result[current_section].append(line[1:].strip())
        
        return result
    
    def analyze_skill_gaps(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Advanced skill gap analysis using Groq.
        
        Args:
            resume_skills: Skills from resume
            job_skills: Required skills from job
            
        Returns:
            Detailed skill gap analysis
        """
        if not self.is_available():
            return {"analysis": "Groq not available for advanced skill analysis"}
        
        prompt = f"""Analyze the skill gap between a candidate and job requirements.

CANDIDATE SKILLS: {', '.join(resume_skills)}
JOB REQUIREMENTS: {', '.join(job_skills)}

Provide analysis in JSON format:
{{
    "critical_gaps": ["skills that are absolutely essential but missing"],
    "minor_gaps": ["skills that would be nice to have but not critical"],
    "transferable_skills": ["candidate skills that could transfer to missing requirements"],
    "learning_path": ["suggested order for learning missing skills"],
    "time_estimate": "estimated time to close major gaps"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a skills assessment expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )
            
            response_text = response.choices[0].message.content.strip()
            return json.loads(response_text)
            
        except Exception as e:
            self.logger.error(f"Skill gap analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_interview_questions(
        self,
        resume_data: ResumeData,
        job_description: JobDescription,
        focus_areas: List[str] = None
    ) -> List[str]:
        """
        Generate targeted interview questions based on resume-job analysis.
        
        Args:
            resume_data: Candidate's resume
            job_description: Job requirements
            focus_areas: Specific areas to focus questions on
            
        Returns:
            List of suggested interview questions
        """
        if not self.is_available():
            return ["Groq not available for question generation"]
        
        focus_text = f"Focus areas: {', '.join(focus_areas)}" if focus_areas else ""
        
        prompt = f"""Generate 5-7 targeted interview questions for this candidate-job match.

JOB: {job_description.job_title} at {job_description.company}
CANDIDATE BACKGROUND: {resume_data.get_skill_names()[:5]}
{focus_text}

Questions should:
1. Assess technical competency
2. Evaluate experience relevance  
3. Identify potential concerns
4. Be specific to this match

Return as a JSON list: ["question1", "question2", ...]"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are an expert interviewer creating targeted questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=512
            )
            
            response_text = response.choices[0].message.content.strip()
            questions = json.loads(response_text)
            return questions if isinstance(questions, list) else [response_text]
            
        except Exception as e:
            self.logger.error(f"Question generation failed: {str(e)}")
            return [f"Error generating questions: {str(e)}"]

