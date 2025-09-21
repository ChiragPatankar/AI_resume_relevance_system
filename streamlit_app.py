"""
Main Streamlit application for the Resume Relevance Check System.

This is the web interface for Innomatics Research Labs placement team
to upload job descriptions, evaluate resumes, and view relevance scores.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from config import config
from models.job_description import JobDescription
from models.resume_data import ResumeData
from models.scoring_result import ScoringResult
from models.comparison_result import ComparisonResult, RankingResult
from parsers.resume_parser import ResumeParser
from processors.text_processor import TextProcessor
from processors.similarity_calculator import SimilarityCalculator

# Page configuration
st.set_page_config(
    page_title="Innomatics Resume Relevance Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2e8b57);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .score-excellent { border-left-color: #28a745; }
    .score-good { border-left-color: #17a2b8; }
    .score-fair { border-left-color: #ffc107; }
    .score-poor { border-left-color: #dc3545; }
    
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'job_description' not in st.session_state:
    st.session_state.job_description = None
if 'resume_results' not in st.session_state:
    st.session_state.resume_results = []
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None

# Initialize components
@st.cache_resource
def load_components():
    """Load and cache the processing components."""
    try:
        resume_parser = ResumeParser()
        text_processor = TextProcessor()
        similarity_calculator = SimilarityCalculator()
        return resume_parser, text_processor, similarity_calculator
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        return None, None, None

@st.cache_resource
def load_hybrid_scorer():
    """Load the real HybridScorer with Groq integration."""
    try:
        from scorers.hybrid_scorer import HybridScorer
        scorer = HybridScorer()
        return scorer
    except Exception as e:
        st.error(f"Failed to load HybridScorer: {str(e)}")
        return None

# Load components
resume_parser, text_processor, similarity_calculator = load_components()

if not all([resume_parser, text_processor, similarity_calculator]):
    st.error("Some components failed to load. The system will run with limited functionality.")
    st.warning("💡 For full functionality, install missing dependencies: `pip install spacy` and `python -m spacy download en_core_web_sm`")
    # Create fallback components
    if not resume_parser:
        from parsers.resume_parser import ResumeParser
        resume_parser = ResumeParser()
    if not text_processor:
        from processors.text_processor import TextProcessor
        text_processor = TextProcessor()
    if not similarity_calculator:
        from processors.similarity_calculator import SimilarityCalculator
        similarity_calculator = SimilarityCalculator()

# Load the real HybridScorer with integrated Groq LLM
hybrid_scorer = load_hybrid_scorer()

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Innomatics Research Labs</h1>
        <h2>Resume Relevance Check System</h2>
        <p>Automated resume evaluation for faster, consistent candidate shortlisting</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with navigation and info."""
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <h3>📊 System Dashboard</h3>
        <p>Placement Team Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🏢 Locations")
    st.sidebar.markdown("- Hyderabad")
    st.sidebar.markdown("- Bangalore") 
    st.sidebar.markdown("- Pune")
    st.sidebar.markdown("- Delhi NCR")
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 📈 Weekly Stats")
    st.sidebar.metric("Job Requirements", "18-20")
    st.sidebar.metric("Applications", "1000+")
    st.sidebar.metric("Processing Time", "< 5 min")
    
    st.sidebar.markdown("---")
    
    # System settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    scoring_threshold = st.sidebar.slider(
        "Minimum Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum relevance score for shortlisting"
    )
    
    max_candidates = st.sidebar.number_input(
        "Max Candidates to Show",
        min_value=5,
        max_value=100,
        value=20,
        help="Maximum number of candidates in results"
    )
    
    # AI Status
    st.sidebar.markdown("### 🤖 AI Enhancement")
    if hybrid_scorer and hasattr(hybrid_scorer, 'groq_enhancer') and hybrid_scorer.groq_enhancer.is_available():
        st.sidebar.success("🚀 Groq LLM Active!")
        st.sidebar.info("Enhanced explanations enabled")
    else:
        st.sidebar.warning("⚠️ LLM enhancement unavailable")
    
    return scoring_threshold, max_candidates

@st.cache_data
def get_sample_job_descriptions():
    """Get sample job descriptions including real JDs from the JD folder."""
    try:
        from utils.jd_loader import jd_loader
        return jd_loader.get_combined_sample_jds()
    except Exception as e:
        logger.warning(f"Could not load JDs from folder: {e}")
        # Fallback to hardcoded samples
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
- Spark/Hadoop for big data""",
                "required_skills": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "Data Analysis", "TensorFlow", "PyTorch"],
                "preferred_skills": ["AWS", "Docker", "Kubernetes", "NLP", "Computer Vision", "Spark"],
                "location": "San Francisco, CA",
                "salary_range": "$120,000 - $180,000"
            }
        }

def render_job_description_input():
    """Render job description input section."""
    st.header("📋 Step 1: Job Description")
    
    # Sample JD selector
    st.markdown("### 🚀 Quick Start with Sample Job Descriptions")
    sample_jds = get_sample_job_descriptions()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_sample = st.selectbox(
            "Choose a sample job description to get started quickly:",
            [""] + list(sample_jds.keys()),
            help="Select a pre-loaded job description for immediate testing"
        )
    
    with col2:
        if st.button("🎯 Load Sample JD", type="primary"):
            if selected_sample and selected_sample in sample_jds:
                sample = sample_jds[selected_sample]
                # Store in session state
                st.session_state.sample_jd_title = sample["title"]
                st.session_state.sample_jd_company = sample["company"]
                st.session_state.sample_jd_description = sample["description"]
                st.success(f"✅ Loaded: {sample['title']}")
                st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Job Description")
        
        # Job basic info with pre-filled values if sample was loaded
        job_title = st.text_input(
            "Job Title", 
            value=st.session_state.get("sample_jd_title", ""),
            placeholder="e.g., Senior Data Scientist"
        )
        company = st.text_input(
            "Company", 
            value=st.session_state.get("sample_jd_company", ""),
            placeholder="e.g., Tech Solutions Inc."
        )
        
        # Job description text
        job_text = st.text_area(
            "Job Description",
            value=st.session_state.get("sample_jd_description", ""),
            height=300,
            placeholder="""
Paste the complete job description here including:
- Role responsibilities
- Required skills and technologies
- Experience requirements
- Educational qualifications
- Preferred qualifications
            """.strip()
        )
        
        # Process job description
        if st.button("📝 Process Job Description", type="primary"):
            if job_text.strip():
                with st.spinner("Processing job description..."):
                    try:
                        job_desc = JobDescription.from_text(
                            job_text, 
                            job_title or "Untitled Position",
                            company or "Unknown Company"
                        )
                        
                        # Extract skills and keywords using text processor
                        skills = text_processor.extract_skills_advanced(job_text)
                        job_desc.required_skills = [skill['name'] for skill in skills[:10]]
                        
                        keywords = text_processor.extract_keywords(job_text, top_k=20)
                        job_desc.keywords = [kw['keyword'] for kw in keywords]
                        
                        st.session_state.job_description = job_desc
                        st.success("✅ Job description processed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error processing job description: {str(e)}")
            else:
                st.warning("Please enter a job description.")
    
    with col2:
        if st.session_state.job_description:
            st.markdown("### ✅ Current Job")
            job = st.session_state.job_description
            
            st.markdown(f"**Title:** {job.job_title}")
            st.markdown(f"**Company:** {job.company or 'Not specified'}")
            
            if job.required_skills:
                st.markdown("**Key Skills:**")
                for skill in job.required_skills[:5]:
                    st.markdown(f"- {skill}")
            
            if st.button("🗑️ Clear Job Description"):
                st.session_state.job_description = None
                st.rerun()

def render_resume_upload():
    """Render resume upload section."""
    st.header("📁 Step 2: Upload Resumes")
    
    if not st.session_state.job_description:
        st.warning("⚠️ Please process a job description first.")
        return
    
    # Info about sample files
    st.info("💡 **Sample data available**: \n- **Resumes**: Check the `Resumes/` folder with 10 real resume PDFs \n- **Job Descriptions**: Real JDs loaded from `JD/` folder plus curated samples")
    
    # Demo option to load sample resumes
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files. Maximum 10MB per file."
        )
    
    with col2:
        st.markdown("**🚀 Quick Demo**")
        if st.button("📁 Load Sample Resumes", type="secondary"):
            # Load sample resumes from Resumes folder
            resumes_folder = Path("Resumes")
            if resumes_folder.exists():
                sample_files = list(resumes_folder.glob("*.pdf"))[:5]  # Load first 5
                if sample_files:
                    st.session_state.demo_files = sample_files
                    st.success(f"✅ Loaded {len(sample_files)} sample resumes!")
                    st.rerun()
                else:
                    st.warning("No PDF files found in Resumes folder")
            else:
                st.warning("Resumes folder not found")
    
    # Handle demo files
    demo_files = st.session_state.get('demo_files', [])
    if demo_files:
        st.info(f"🎯 **Demo Mode**: {len(demo_files)} sample resumes loaded from Resumes folder")
        uploaded_files = demo_files
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")
        
        # Process uploaded files
        if st.button("🚀 Process Resumes", type="primary"):
            if len(uploaded_files) > 50:
                st.warning("Maximum 50 files allowed at once.")
                return
            
            # Use the real HybridScorer with integrated Groq
            if not hybrid_scorer:
                st.error("❌ Scoring system not available")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Handle both uploaded files and demo files (Path objects)
                    if isinstance(uploaded_file, Path):
                        # Demo file from Resumes folder
                        file_name = uploaded_file.name
                        file_path = str(uploaded_file)
                        status_text.text(f"Processing {file_name} (demo)...")
                        
                        # Parse resume directly from file path
                        resume_data = resume_parser.parse_resume(file_path)
                    else:
                        # Regular uploaded file
                        file_name = uploaded_file.name
                        status_text.text(f"Processing {file_name}...")
                        
                        # Save uploaded file temporarily
                        temp_path = f"temp/{file_name}"
                        os.makedirs("temp", exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Parse resume
                        resume_data = resume_parser.parse_resume(temp_path)
                        file_path = temp_path
                    
                    if resume_data and resume_data.is_valid():
                        # Score resume using HybridScorer with Groq
                        scoring_result = hybrid_scorer.score_resume(
                            resume_data, 
                            st.session_state.job_description
                        )
                        
                        results.append({
                            'resume_data': resume_data,
                            'scoring_result': scoring_result
                        })
                    else:
                        st.warning(f"Could not process {file_name} - invalid or corrupted file")
                    
                    # Clean up temp file (only for uploaded files, not demo files)
                    if not isinstance(uploaded_file, Path) and 'temp_path' in locals() and os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
            
            # Store results
            st.session_state.resume_results = results
            
            # Create comparison result
            if results:
                rankings = []
                for i, result in enumerate(results):
                    ranking = RankingResult(
                        resume_filename=result['resume_data'].file_name,
                        candidate_name=result['resume_data'].contact_info.name,
                        scoring_result=result['scoring_result']
                    )
                    rankings.append(ranking)
                
                comparison = ComparisonResult(
                    job_title=st.session_state.job_description.job_title,
                    job_description_summary=st.session_state.job_description.summary or "No summary",
                    rankings=rankings
                )
                
                st.session_state.comparison_result = comparison
                
            status_text.text("✅ Processing complete!")
            progress_bar.progress(1.0)
            
            st.success(f"Successfully processed {len(results)} resumes!")

def render_results():
    """Render results section."""
    if not st.session_state.resume_results:
        return
    
    st.header("📊 Step 3: Results & Analysis")
    
    results = st.session_state.resume_results
    comparison = st.session_state.comparison_result
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(results))
    
    with col2:
        avg_score = sum(r['scoring_result'].overall_score for r in results) / len(results)
        st.metric("Average Score", f"{avg_score:.2f}")
    
    with col3:
        high_score_count = sum(1 for r in results if r['scoring_result'].overall_score >= 0.7)
        st.metric("High Scores (≥70%)", high_score_count)
    
    with col4:
        best_score = max(r['scoring_result'].overall_score for r in results)
        st.metric("Best Score", f"{best_score:.2f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📈 Analytics", "🔍 Detailed View", "📥 Export"])
    
    with tab1:
        render_rankings_tab(results)
    
    with tab2:
        render_analytics_tab(results, comparison)
    
    with tab3:
        render_detailed_view_tab(results)
    
    with tab4:
        render_export_tab(results, comparison)

def render_rankings_tab(results):
    """Render rankings tab."""
    st.subheader("🏆 Candidate Rankings")
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['scoring_result'].overall_score, reverse=True)
    
    # Create rankings table
    rankings_data = []
    for i, result in enumerate(sorted_results[:20]):  # Top 20
        scoring = result['scoring_result']
        contact = result['resume_data'].contact_info
        
        rankings_data.append({
            'Rank': i + 1,
            'Candidate': contact.name or 'Unknown',
            'Filename': result['resume_data'].file_name,
            'Overall Score': f"{scoring.overall_score:.3f}",
            'Score %': f"{int(scoring.overall_score * 100)}%",
            'Category': scoring.fit_level,
            'Skills': f"{scoring.breakdown.skills_score:.2f}",
            'Experience': f"{scoring.breakdown.experience_score:.2f}",
            'Education': f"{scoring.breakdown.education_score:.2f}",
            'Confidence': f"{scoring.confidence_score:.2f}",
            'Email': contact.email or 'Not provided'
        })
    
    df = pd.DataFrame(rankings_data)
    
    # Display table with highlighting
    def highlight_scores(val):
        if 'Score %' in val.name:
            score = float(val.replace('%', '')) / 100
            if score >= 0.8:
                return ['background-color: #d4edda'] * len(val)
            elif score >= 0.7:
                return ['background-color: #d1ecf1'] * len(val)
            elif score >= 0.6:
                return ['background-color: #fff3cd'] * len(val)
            else:
                return ['background-color: #f8d7da'] * len(val)
        return [''] * len(val)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Top candidates details
    st.subheader("🌟 Top 3 Candidates")
    
    for i, result in enumerate(sorted_results[:3]):
        with st.expander(f"#{i+1} - {result['resume_data'].contact_info.name or 'Unknown'} ({int(result['scoring_result'].overall_score * 100)}%)"):
            render_candidate_summary(result)

def render_candidate_summary(result):
    """Render individual candidate summary."""
    resume_data = result['resume_data']
    scoring = result['scoring_result']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📊 Scores**")
        st.markdown(f"- Overall: {int(scoring.overall_score * 100)}%")
        st.markdown(f"- Skills: {int(scoring.breakdown.skills_score * 100)}%")
        st.markdown(f"- Experience: {int(scoring.breakdown.experience_score * 100)}%")
        st.markdown(f"- Education: {int(scoring.breakdown.education_score * 100)}%")
        
        st.markdown("**👤 Contact**")
        if resume_data.contact_info.email:
            st.markdown(f"- Email: {resume_data.contact_info.email}")
        if resume_data.contact_info.phone:
            st.markdown(f"- Phone: {resume_data.contact_info.phone}")
    
    with col2:
        st.markdown("**✅ Matched Skills**")
        for skill in scoring.breakdown.matched_skills[:5]:
            st.markdown(f"- {skill}")
        
        st.markdown("**❌ Missing Skills**")
        for skill in scoring.breakdown.missing_skills[:3]:
            st.markdown(f"- {skill}")
    
    if scoring.explanation and hasattr(scoring.explanation, 'strengths') and scoring.explanation.strengths:
        st.markdown("**💪 Strengths**")
        for strength in scoring.explanation.strengths[:3]:
            st.markdown(f"- {strength}")

def render_analytics_tab(results, comparison):
    """Render analytics tab."""
    st.subheader("📈 Advanced Analytics Dashboard")
    
    if not results:
        st.info("No data available for analytics.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    scores = [r['scoring_result'].overall_score for r in results]
    
    with col1:
        avg_score = sum(scores) / len(scores)
        st.metric(
            "Average Score", 
            f"{avg_score:.1%}",
            delta=f"{(avg_score - 0.5):.1%}" if avg_score > 0.5 else f"{(avg_score - 0.5):.1%}"
        )
    
    with col2:
        high_performers = len([s for s in scores if s >= 0.8])
        st.metric("High Performers", f"{high_performers}/{len(scores)}", f"{high_performers/len(scores):.1%}")
    
    with col3:
        best_score = max(scores)
        st.metric("Best Score", f"{best_score:.1%}", "🏆")
    
    with col4:
        score_std = pd.Series(scores).std()
        st.metric("Score Spread", f"{score_std:.3f}", "📊")
    
    st.markdown("---")
    
    # Enhanced score distribution with overlays
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Score Distribution Analysis**")
        
        # Create advanced histogram with density curve
        fig_hist = go.Figure()
        
        # Add histogram
        fig_hist.add_trace(go.Histogram(
            x=scores,
            nbinsx=15,
            name="Score Distribution",
            marker=dict(
                color='rgba(26, 118, 255, 0.7)',
                line=dict(color='rgba(26, 118, 255, 1.0)', width=2)
            ),
            opacity=0.7
        ))
        
        # Add average line
        fig_hist.add_vline(
            x=avg_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_score:.1%}",
            annotation_position="top right"
        )
        
        # Add threshold lines
        fig_hist.add_vline(x=0.8, line_dash="dot", line_color="green", annotation_text="High (80%)")
        fig_hist.add_vline(x=0.6, line_dash="dot", line_color="orange", annotation_text="Medium (60%)")
        
        fig_hist.update_layout(
            title="Resume Score Distribution with Benchmarks",
            xaxis_title="Relevance Score",
            yaxis_title="Number of Candidates",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Performance Categories**")
        
        # Enhanced pie chart with custom colors
        categories = [r['scoring_result'].fit_level for r in results]
        category_counts = pd.Series(categories).value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hole=.4,
            marker=dict(colors=colors[:len(category_counts)], line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent+value',
            textfont_size=12
        )])
        
        fig_pie.update_layout(
            title="Candidate Performance Categories",
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Component scores comparison with multiple visualizations
    st.markdown("---")
    st.markdown("**🎯 Component Score Analysis**")
    
    # Prepare component data
    component_data = []
    for i, result in enumerate(results):
        scoring = result['scoring_result']
        contact = result['resume_data'].contact_info
        component_data.append({
            'Rank': i + 1,
            'Candidate': contact.name or f"Candidate {i+1}",
            'Skills': scoring.breakdown.skills_score,
            'Experience': scoring.breakdown.experience_score,
            'Education': scoring.breakdown.education_score,
            'Keywords': scoring.breakdown.keywords_score,
            'Overall': scoring.overall_score
        })
    
    df_components = pd.DataFrame(component_data)
    
    # Three visualization tabs
    tab1, tab2, tab3 = st.tabs(["🕸️ Radar Analysis", "📊 Component Bars", "🎯 Score Matrix"])
    
    with tab1:
        st.markdown("**Top 5 Candidates - Radar Chart**")
        
        fig_radar = go.Figure()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (_, row) in enumerate(df_components.head(5).iterrows()):
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Skills'], row['Experience'], row['Education'], row['Keywords']],
                theta=['Skills', 'Experience', 'Education', 'Keywords'],
                fill='toself',
                name=f"#{row['Rank']} {row['Candidate'][:10]}",
                line=dict(color=colors[i % len(colors)]),
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [0.1])}"
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.1%'
                )
            ),
            showlegend=True,
            title="Multi-Dimensional Skills Assessment",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        st.markdown("**Component Score Comparison (All Candidates)**")
        
        # Create stacked bar chart
        fig_bar = go.Figure()
        
        candidates = [f"#{row['Rank']}" for _, row in df_components.iterrows()]
        
        fig_bar.add_trace(go.Bar(
            name='Skills',
            x=candidates,
            y=df_components['Skills'],
            marker_color='#FF6B6B',
            text=[f"{val:.1%}" for val in df_components['Skills']],
            textposition='inside'
        ))
        
        fig_bar.add_trace(go.Bar(
            name='Experience',
            x=candidates,
            y=df_components['Experience'],
            marker_color='#4ECDC4',
            text=[f"{val:.1%}" for val in df_components['Experience']],
            textposition='inside'
        ))
        
        fig_bar.add_trace(go.Bar(
            name='Education',
            x=candidates,
            y=df_components['Education'],
            marker_color='#45B7D1',
            text=[f"{val:.1%}" for val in df_components['Education']],
            textposition='inside'
        ))
        
        fig_bar.add_trace(go.Bar(
            name='Keywords',
            x=candidates,
            y=df_components['Keywords'],
            marker_color='#96CEB4',
            text=[f"{val:.1%}" for val in df_components['Keywords']],
            textposition='inside'
        ))
        
        fig_bar.update_layout(
            title="Component Scores by Candidate",
            xaxis_title="Candidate Rank",
            yaxis_title="Score",
            barmode='group',
            height=500,
            yaxis=dict(tickformat='.1%'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.markdown("**Score Heatmap Matrix**")
        
        # Create heatmap
        heatmap_data = df_components[['Skills', 'Experience', 'Education', 'Keywords']].head(15)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Skills', 'Experience', 'Education', 'Keywords'],
            y=[f"#{i+1}" for i in range(len(heatmap_data))],
            colorscale='RdYlBu_r',
            text=[[f"{val:.1%}" for val in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                title="Score",
                tickformat='.1%'
            )
        ))
        
        fig_heatmap.update_layout(
            title="Score Matrix - Top 15 Candidates",
            xaxis_title="Assessment Categories",
            yaxis_title="Candidate Rank",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Enhanced skills gap analysis
    st.markdown("---")
    st.markdown("**🎯 Advanced Skills Gap Analysis**")
    
    # Collect skill data
    all_missing_skills = []
    all_matched_skills = []
    
    for result in results:
        all_missing_skills.extend(result['scoring_result'].breakdown.missing_skills)
        all_matched_skills.extend(result['scoring_result'].breakdown.matched_skills)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Most Common Missing Skills**")
        
        if all_missing_skills:
            missing_skills_counts = pd.Series(all_missing_skills).value_counts().head(12)
            
            fig_missing = go.Figure(go.Bar(
                x=missing_skills_counts.values,
                y=missing_skills_counts.index,
                orientation='h',
                marker=dict(
                    color=missing_skills_counts.values,
                    colorscale='Reds',
                    colorbar=dict(title="Gap Frequency")
                ),
                text=[f"{val} candidates" for val in missing_skills_counts.values],
                textposition='auto'
            ))
            
            fig_missing.update_layout(
                title="Critical Skills Gaps",
                xaxis_title="Number of Candidates Missing Skill",
                yaxis_title="Skills",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.info("No missing skills data available")
    
    with col2:
        st.markdown("**🟢 Most Common Matched Skills**")
        
        if all_matched_skills:
            matched_skills_counts = pd.Series(all_matched_skills).value_counts().head(12)
            
            fig_matched = go.Figure(go.Bar(
                x=matched_skills_counts.values,
                y=matched_skills_counts.index,
                orientation='h',
                marker=dict(
                    color=matched_skills_counts.values,
                    colorscale='Greens',
                    colorbar=dict(title="Match Frequency")
                ),
                text=[f"{val} candidates" for val in matched_skills_counts.values],
                textposition='auto'
            ))
            
            fig_matched.update_layout(
                title="Commonly Found Skills",
                xaxis_title="Number of Candidates with Skill",
                yaxis_title="Skills",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_matched, use_container_width=True)
        else:
            st.info("No matched skills data available")
    
    # Skills coverage analysis
    if all_missing_skills and all_matched_skills:
        st.markdown("**📊 Skills Coverage Overview**")
        
        # Calculate skill coverage percentages
        all_skills = list(set(all_missing_skills + all_matched_skills))
        coverage_data = []
        
        for skill in all_skills[:20]:  # Top 20 skills
            matched_count = all_matched_skills.count(skill)
            missing_count = all_missing_skills.count(skill)
            total_mentions = matched_count + missing_count
            
            if total_mentions > 0:
                coverage_percentage = (matched_count / total_mentions) * 100
                coverage_data.append({
                    'Skill': skill,
                    'Coverage': coverage_percentage,
                    'Matched': matched_count,
                    'Missing': missing_count,
                    'Total': total_mentions
                })
        
        if coverage_data:
            df_coverage = pd.DataFrame(coverage_data).sort_values('Coverage', ascending=True)
            
            fig_coverage = go.Figure()
            
            # Add coverage bars
            fig_coverage.add_trace(go.Bar(
                name='Coverage %',
                x=df_coverage['Coverage'],
                y=df_coverage['Skill'],
                orientation='h',
                marker=dict(
                    color=df_coverage['Coverage'],
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100,
                    colorbar=dict(title="Coverage %")
                ),
                text=[f"{val:.0f}%" for val in df_coverage['Coverage']],
                textposition='auto'
            ))
            
            fig_coverage.update_layout(
                title="Skills Coverage Analysis (% of candidates who have each skill)",
                xaxis_title="Coverage Percentage",
                yaxis_title="Skills",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Show coverage summary
            avg_coverage = df_coverage['Coverage'].mean()
            low_coverage_skills = len(df_coverage[df_coverage['Coverage'] < 30])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Skills Coverage", f"{avg_coverage:.1f}%")
            with col2:
                st.metric("Low Coverage Skills (<30%)", low_coverage_skills)
            with col3:
                st.metric("Skills Analyzed", len(df_coverage))

def render_detailed_view_tab(results):
    """Render detailed view tab."""
    st.subheader("🔍 Detailed Candidate Analysis")
    
    if not results:
        st.info("No candidates to display.")
        return
    
    # Candidate selector
    candidate_names = [f"{i+1}. {r['resume_data'].contact_info.name or r['resume_data'].file_name}" 
                      for i, r in enumerate(results)]
    
    selected_idx = st.selectbox(
        "Select Candidate for Detailed View",
        range(len(candidate_names)),
        format_func=lambda x: candidate_names[x]
    )
    
    if selected_idx is not None:
        result = results[selected_idx]
        resume_data = result['resume_data']
        scoring = result['scoring_result']
        
        # Candidate overview
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📊 Overall Score")
            score_color = (
                "🟢" if scoring.overall_score >= 0.8 else
                "🟡" if scoring.overall_score >= 0.6 else "🔴"
            )
            st.markdown(f"## {score_color} {int(scoring.overall_score * 100)}%")
            st.markdown(f"**Category:** {scoring.fit_level}")
            st.markdown(f"**Confidence:** {int(scoring.confidence_score * 100)}%")
        
        with col2:
            st.markdown("### 👤 Contact Information")
            if resume_data.contact_info.name:
                st.markdown(f"**Name:** {resume_data.contact_info.name}")
            if resume_data.contact_info.email:
                st.markdown(f"**Email:** {resume_data.contact_info.email}")
            if resume_data.contact_info.phone:
                st.markdown(f"**Phone:** {resume_data.contact_info.phone}")
            if resume_data.contact_info.linkedin:
                st.markdown(f"**LinkedIn:** {resume_data.contact_info.linkedin}")
        
        with col3:
            st.markdown("### 📈 Component Scores")
            st.markdown(f"**Skills:** {int(scoring.breakdown.skills_score * 100)}%")
            st.markdown(f"**Experience:** {int(scoring.breakdown.experience_score * 100)}%")
            st.markdown(f"**Education:** {int(scoring.breakdown.education_score * 100)}%")
            st.markdown(f"**Keywords:** {int(scoring.breakdown.keywords_score * 100)}%")
        
        st.markdown("---")
        
        # Detailed sections
        tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Skills Analysis", "💼 Experience", "🎓 Education", "💡 Recommendations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Matched Skills")
                
                # Clean matched skills to prevent overlaps
                matched_skills_clean = [
                    skill for skill in scoring.breakdown.matched_skills 
                    if skill not in scoring.breakdown.missing_skills
                ]
                
                for skill in matched_skills_clean:
                    st.markdown(f"- ✅ {skill}")
                
                if not matched_skills_clean:
                    st.info("No matched skills identified")
            
            with col2:
                st.markdown("#### ❌ Missing Skills")
                
                # Clean missing skills to prevent overlaps  
                missing_skills_clean = [
                    skill for skill in scoring.breakdown.missing_skills
                    if skill not in scoring.breakdown.matched_skills
                ]
                
                for skill in missing_skills_clean:
                    st.markdown(f"- ❌ {skill}")
                
                if not missing_skills_clean:
                    st.success("No critical skills missing!")
        
        with tab2:
            st.markdown("#### Work Experience")
            for exp in resume_data.experience:
                with st.expander(f"{exp.job_title or 'Position'} at {exp.company or 'Company'}"):
                    if exp.start_date or exp.end_date:
                        duration = f"{exp.start_date or 'Unknown'} - {exp.end_date or 'Current'}"
                        st.markdown(f"**Duration:** {duration}")
                    
                    if exp.description:
                        st.markdown(f"**Description:** {exp.description}")
                    
                    if exp.responsibilities:
                        st.markdown("**Responsibilities:**")
                        for resp in exp.responsibilities[:3]:
                            st.markdown(f"- {resp}")
        
        with tab3:
            st.markdown("#### Education Background")
            for edu in resume_data.education:
                with st.expander(f"{edu.degree or 'Degree'} - {edu.institution or 'Institution'}"):
                    if edu.field_of_study:
                        st.markdown(f"**Field:** {edu.field_of_study}")
                    if edu.start_date or edu.end_date:
                        duration = f"{edu.start_date or 'Unknown'} - {edu.end_date or 'Unknown'}"
                        st.markdown(f"**Duration:** {duration}")
                    if edu.gpa:
                        st.markdown(f"**GPA:** {edu.gpa}")
        
        with tab4:
            st.markdown("#### 💪 Strengths")
            if scoring.explanation and hasattr(scoring.explanation, 'strengths') and scoring.explanation.strengths:
                for strength in scoring.explanation.strengths:
                    st.markdown(f"- ✅ {strength}")
            else:
                st.info("No specific strengths identified in analysis")
            
            st.markdown("#### 🎯 Areas for Improvement")
            if scoring.explanation and hasattr(scoring.explanation, 'weaknesses') and scoring.explanation.weaknesses:
                for weakness in scoring.explanation.weaknesses:
                    st.markdown(f"- 🔍 {weakness}")
            else:
                st.info("No specific weaknesses identified in analysis")
            
            st.markdown("#### 📝 Recommendations")
            if scoring.explanation and hasattr(scoring.explanation, 'recommendations') and scoring.explanation.recommendations:
                for rec in scoring.explanation.recommendations:
                    st.markdown(f"- 💡 {rec}")
            else:
                st.info("No specific recommendations available")
        
        # Add individual candidate analytics section
        st.markdown("---")
        st.markdown("### 📊 Individual Candidate Analytics")
        
        # Create detailed analytics for this candidate
        render_individual_candidate_analytics(resume_data, scoring)

def render_individual_candidate_analytics(resume_data, scoring):
    """Render detailed analytics for an individual candidate."""
    
    # Create analytics tabs
    analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
        "📊 Score Breakdown", "🎯 Skills Analysis", "📈 Experience Analysis", "🏆 Competitive Analysis"
    ])
    
    with analytics_tab1:
        st.markdown("#### 📊 Detailed Score Breakdown")
        
        # Create score breakdown visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Score breakdown chart
            categories = ['Skills', 'Experience', 'Education', 'Keywords']
            scores = [
                scoring.breakdown.skills_score * 100,
                scoring.breakdown.experience_score * 100,
                scoring.breakdown.education_score * 100,
                scoring.breakdown.keywords_score * 100
            ]
            
            fig_individual_bar = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                    text=[f"{score:.1f}%" for score in scores],
                    textposition='auto'
                )
            ])
            
            fig_individual_bar.update_layout(
                title="Component Score Breakdown",
                yaxis_title="Score (%)",
                xaxis_title="Assessment Categories",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_individual_bar, use_container_width=True)
        
        with col2:
            # Radar chart for this candidate
            fig_radar_individual = go.Figure()
            
            fig_radar_individual.add_trace(go.Scatterpolar(
                r=[score/100 for score in scores],
                theta=categories,
                fill='toself',
                name=resume_data.contact_info.name or "Candidate",
                line=dict(color='#FF6B6B'),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig_radar_individual.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat='.1%'
                    )
                ),
                title="Multi-Dimensional Assessment Profile",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_radar_individual, use_container_width=True)
        
        # Score metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{scoring.overall_score:.1%}", f"{scoring.fit_level}")
        with col2:
            st.metric("Confidence", f"{scoring.confidence_score:.1%}")
        with col3:
            best_category = categories[scores.index(max(scores))]
            st.metric("Strongest Area", best_category, f"{max(scores):.1f}%")
        with col4:
            weakest_category = categories[scores.index(min(scores))]
            st.metric("Improvement Area", weakest_category, f"{min(scores):.1f}%")
    
    with analytics_tab2:
        st.markdown("#### 🎯 Skills Analysis Dashboard")
        
        # Skills comparison
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**✅ Matched Skills Analysis**")
            if scoring.breakdown.matched_skills:
                matched_skills_data = {
                    'Skill': scoring.breakdown.matched_skills[:10],
                    'Status': ['✅ Matched'] * len(scoring.breakdown.matched_skills[:10])
                }
                
                fig_matched = px.bar(
                    y=matched_skills_data['Skill'],
                    x=[1] * len(matched_skills_data['Skill']),
                    orientation='h',
                    title="Skills Portfolio - Strengths",
                    color_discrete_sequence=['#28a745']
                )
                fig_matched.update_layout(height=400, showlegend=False, xaxis_title="Skill Present")
                st.plotly_chart(fig_matched, use_container_width=True)
            else:
                st.info("No matched skills identified")
        
        with col2:
            st.markdown("**❌ Missing Skills Analysis**")
            if scoring.breakdown.missing_skills:
                missing_skills_data = {
                    'Skill': scoring.breakdown.missing_skills[:10],
                    'Impact': ['❌ Missing'] * len(scoring.breakdown.missing_skills[:10])
                }
                
                fig_missing = px.bar(
                    y=missing_skills_data['Skill'],
                    x=[1] * len(missing_skills_data['Skill']),
                    orientation='h',
                    title="Skills Development Opportunities",
                    color_discrete_sequence=['#dc3545']
                )
                fig_missing.update_layout(height=400, showlegend=False, xaxis_title="Skill Gap")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No critical skills missing!")
        
        # Skills coverage analysis
        total_skills_evaluated = len(scoring.breakdown.matched_skills) + len(scoring.breakdown.missing_skills)
        if total_skills_evaluated > 0:
            coverage_percentage = (len(scoring.breakdown.matched_skills) / total_skills_evaluated) * 100
            
            st.markdown("**📊 Skills Coverage Summary**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skills Coverage", f"{coverage_percentage:.1f}%")
            with col2:
                st.metric("Skills Matched", len(scoring.breakdown.matched_skills))
            with col3:
                st.metric("Skills to Develop", len(scoring.breakdown.missing_skills))
    
    with analytics_tab3:
        st.markdown("#### 📈 Experience & Education Analysis")
        
        # Experience analysis
        if resume_data.experience:
            st.markdown("**💼 Professional Experience Analysis**")
            
            # Create experience data for visualization
            exp_data = []
            for i, exp in enumerate(resume_data.experience[:7]):  # Show top 7
                duration_text = f"{exp.start_date or 'Unknown'} - {exp.end_date or 'Current'}"
                exp_data.append({
                    'Position': exp.job_title or f"Position {i+1}",
                    'Company': exp.company or "Company Not Specified",
                    'Duration': duration_text,
                    'Index': i + 1,
                    'Description': exp.description[:100] + "..." if exp.description and len(exp.description) > 100 else exp.description or "No description available"
                })
            
            # Create experience bar chart
            if exp_data:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Experience positions chart
                    fig_exp = px.bar(
                        pd.DataFrame(exp_data),
                        x='Index',
                        y='Position',
                        color='Company',
                        orientation='h',
                        title="Career Experience Overview",
                        labels={'Index': 'Experience #', 'Position': 'Job Positions'}
                    )
                    fig_exp.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig_exp, use_container_width=True)
                
                with col2:
                    st.markdown("**📋 Experience Details**")
                    for exp in exp_data:
                        with st.expander(f"**{exp['Position']}**"):
                            st.markdown(f"**Company:** {exp['Company']}")
                            st.markdown(f"**Duration:** {exp['Duration']}")
                            st.markdown(f"**Description:** {exp['Description']}")
            else:
                st.info("No experience data available for visualization")
        else:
            st.info("No professional experience information found")
        
        # Education analysis
        if resume_data.education:
            st.markdown("**🎓 Educational Background Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                education_levels = [edu.degree or "Degree" for edu in resume_data.education]
                education_counts = pd.Series(education_levels).value_counts()
                
                fig_education = px.pie(
                    values=education_counts.values,
                    names=education_counts.index,
                    title="Educational Qualifications Distribution"
                )
                st.plotly_chart(fig_education, use_container_width=True)
            
            with col2:
                st.markdown("**📚 Education Details**")
                for edu in resume_data.education:
                    st.markdown(f"**{edu.degree or 'Degree'}**")
                    if edu.institution:
                        st.markdown(f"- Institution: {edu.institution}")
                    if edu.field_of_study:
                        st.markdown(f"- Field: {edu.field_of_study}")
                    if edu.gpa:
                        st.markdown(f"- GPA: {edu.gpa}")
                    st.markdown("---")
    
    with analytics_tab4:
        st.markdown("#### 🏆 Competitive Positioning Analysis")
        
        # Compare against average (simulated for demo)
        st.markdown("**📊 Benchmarking Against Pool Average**")
        
        # Simulated benchmark data (in real implementation, this would come from all candidates)
        benchmark_scores = {
            'Skills': 0.3,  # 30% average
            'Experience': 0.25,  # 25% average
            'Education': 0.6,  # 60% average
            'Keywords': 0.35  # 35% average
        }
        
        candidate_scores = {
            'Skills': scoring.breakdown.skills_score,
            'Experience': scoring.breakdown.experience_score,
            'Education': scoring.breakdown.education_score,
            'Keywords': scoring.breakdown.keywords_score
        }
        
        # Create comparison chart
        comparison_data = []
        for category in benchmark_scores.keys():
            comparison_data.extend([
                {'Category': category, 'Type': 'Pool Average', 'Score': benchmark_scores[category] * 100},
                {'Category': category, 'Type': 'This Candidate', 'Score': candidate_scores[category] * 100}
            ])
        
        fig_comparison = px.bar(
            pd.DataFrame(comparison_data),
            x='Category',
            y='Score',
            color='Type',
            barmode='group',
            title="Performance vs. Candidate Pool Average",
            color_discrete_map={
                'Pool Average': '#cccccc',
                'This Candidate': '#FF6B6B'
            }
        )
        fig_comparison.update_layout(height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Competitive insights
        st.markdown("**🎯 Competitive Insights**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**💪 Competitive Advantages:**")
            for category, score in candidate_scores.items():
                if score > benchmark_scores[category]:
                    improvement = ((score - benchmark_scores[category]) / benchmark_scores[category]) * 100
                    st.markdown(f"- ✅ **{category}**: {improvement:.1f}% above average")
        
        with col2:
            st.markdown("**🎯 Areas for Development:**")
            for category, score in candidate_scores.items():
                if score < benchmark_scores[category]:
                    gap = ((benchmark_scores[category] - score) / benchmark_scores[category]) * 100
                    st.markdown(f"- 🔍 **{category}**: {gap:.1f}% below average")
        
        # Overall positioning
        overall_vs_benchmark = scoring.overall_score - 0.35  # Assumed pool average of 35%
        if overall_vs_benchmark > 0:
            st.success(f"🏆 This candidate scores {overall_vs_benchmark:.1%} above the pool average!")
        else:
            st.warning(f"📈 This candidate scores {abs(overall_vs_benchmark):.1%} below the pool average")

def render_export_tab(results, comparison):
    """Render export tab."""
    st.subheader("📥 Export Results")
    
    if not results:
        st.info("No data to export.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Export Rankings CSV")
        
        if comparison:
            # Create export data
            export_data = comparison.export_summary_table()
            df_export = pd.DataFrame(export_data)
            
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Rankings CSV",
                data=csv,
                file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.markdown("**Preview:**")
            st.dataframe(df_export.head(), use_container_width=True)
    
    with col2:
        st.markdown("### 📄 Export Detailed Report")
        
        # Create detailed JSON report
        report_data = {
            "job_description": {
                "title": st.session_state.job_description.job_title,
                "company": st.session_state.job_description.company,
                "processed_date": datetime.now().isoformat()
            },
            "summary": {
                "total_resumes": len(results),
                "average_score": sum(r['scoring_result'].overall_score for r in results) / len(results),
                "high_performers": len([r for r in results if r['scoring_result'].overall_score >= 0.7])
            },
            "candidates": [
                {
                    "rank": i + 1,
                    "filename": r['resume_data'].file_name,
                    "name": r['resume_data'].contact_info.name,
                    "email": r['resume_data'].contact_info.email,
                    "overall_score": r['scoring_result'].overall_score,
                    "category": r['scoring_result'].fit_level,
                    "matched_skills": r['scoring_result'].breakdown.matched_skills,
                    "missing_skills": r['scoring_result'].breakdown.missing_skills
                }
                for i, r in enumerate(sorted(results, key=lambda x: x['scoring_result'].overall_score, reverse=True))
            ]
        }
        
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📥 Download Detailed Report (JSON)",
            data=report_json,
            file_name=f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main application function."""
    render_header()
    
    # Sidebar
    scoring_threshold, max_candidates = render_sidebar()
    
    # Main content
    render_job_description_input()
    
    st.markdown("---")
    
    render_resume_upload()
    
    st.markdown("---")
    
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎓 <strong>Innomatics Research Labs</strong> - Resume Relevance Check System</p>
        <p>Automated recruitment solution for faster, consistent candidate evaluation</p>
        <p><em>Serving Hyderabad • Bangalore • Pune • Delhi NCR</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
