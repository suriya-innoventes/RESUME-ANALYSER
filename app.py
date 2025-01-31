import streamlit as st 
import json
import re
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import io
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


ESSENTIAL_SECTIONS = {
    'Work Experience': {
        'weight': 30,
        'criteria': ['duration_match', 'responsibility_match', 'role_relevance'],
        'analysis_prompt': "Analyze job duration, responsibility alignment, and role relevance compared to JD requirements."
    },
    'Skills': {
        'weight': 35,
        'criteria': ['technical_skills', 'tools_match', 'certifications'],
        'analysis_prompt': "Evaluate required technical competencies, tool proficiency, and certifications."
    },
    'Education': {
        'weight': 10,
        'criteria': ['degree_match', 'institution_quality', 'academic_recognition'],
        'analysis_prompt': "Verify required degrees, institution reputation, and academic achievements."
    },
    'Requirements': {
        'weight': 25,
        'criteria': ['experience_years', 'location', 'security_clearance'],
        'analysis_prompt': "Validate minimum experience, location flexibility, and clearance status."
    }
}

def safe_read_file(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(io.BytesIO(file.read()))
            return "\n".join(para.text for para in doc.paragraphs)
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        return ""
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

def validate_json_response(response):
    try:
        parsed = json.loads(response)
        required_keys = ['score', 'matches', 'missing', 'feedback']
        if all(key in parsed for key in required_keys):
            return parsed
        return None
    except json.JSONDecodeError:
        return None

def analyze_section_with_llm(jd_context, resume_text, section):
    client = Groq(api_key="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")
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": f"""
                    Perform detailed analysis of resume section against job requirements.
                    Focus on: {ESSENTIAL_SECTIONS[section]['analysis_prompt']}
                    Provide JSON response with:
                    {{
                        "score": 0-100,
                        "matches": ["top 3 matches"],
                        "missing": ["top 3 gaps"],
                        "feedback": "concise professional assessment"
                    }}
                    """},
                    {"role": "user", "content": f"Job Description: {jd_context[:3000]}\nResume Content: {resume_text[:3000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=400
            )
            
            content = response.choices[0].message.content
            parsed = validate_json_response(content)
            
            if parsed:
                return parsed
            raise ValueError("Invalid response format")
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff_factor * attempt)
            else:
                return {"score": 0, "matches": [], "missing": [], "feedback": f"Analysis failed: {str(e)}"}
    
    return {"score": 0, "matches": [], "missing": [], "feedback": "API error"}

def analyze_resume(jd_text, resume_text):
    analysis = {
        "total_score": 0,
        "is_qualified": False,
        "section_details": {},
        "recommendation": ""
    }
    
    try:
        # Base text similarity analysis
        vectorizer = TfidfVectorizer(stop_words='english')
        base_score = cosine_similarity(
            vectorizer.fit_transform([jd_text]),
            vectorizer.transform([resume_text])
        )[0][0] * 100
        
        # Section-wise detailed analysis
        total_weight = sum(s['weight'] for s in ESSENTIAL_SECTIONS.values())
        weighted_score = 0
        
        for section, config in ESSENTIAL_SECTIONS.items():
            result = analyze_section_with_llm(jd_text, resume_text, section)
            section_weight = config['weight'] / total_weight
            analysis["section_details"][section] = {
                "score": result["score"],
                "matches": result["matches"][:3],
                "missing": result["missing"][:3],
                "feedback": result["feedback"]
            }
            weighted_score += result["score"] * section_weight
            time.sleep(1.5)  # Rate limit protection
        
        # Combine scores with business logic
        analysis["total_score"] = round(0.7 * weighted_score + 0.3 * base_score, 1)
        analysis["is_qualified"] = analysis["total_score"] >= 75
        analysis["recommendation"] = "Strong Candidate" if analysis["total_score"] >= 85 else \
                                    "Qualified" if analysis["is_qualified"] else "Not Qualified"
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    
    return analysis

def main():
    st.set_page_config(page_title="PROJECT ATS", layout="wide")
    
    # Initialize session state
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = []
    
    # Job Description Processing
    if 'jd_text' not in st.session_state:
        st.title("📋 Job Description Upload")
        jd_file = st.file_uploader("Upload Job Description Document", type=["pdf", "docx", "txt"])
        
        if jd_file:
            with st.spinner("Extracting Job Requirements..."):
                jd_text = safe_read_file(jd_file)
                if jd_text:
                    st.session_state.jd_text = jd_text
                    st.success("JD Processed Successfully!")
                    
                    with st.expander("Preview Key Requirements"):
                        st.write(jd_text[:1500] + "...")
                    
                    if st.button("Proceed to Resume Screening"):
                        st.rerun()
    
    # Resume Analysis Section
    else:
        st.title("📄 Resume Evaluation")
        
        st.subheader("Upload Candidate Resumes")
        resume_files = st.file_uploader("Select Resume Files", 
                                      type=["pdf", "docx", "txt"],
                                      accept_multiple_files=True)
        
        if resume_files and st.button("Initiate Comprehensive Analysis"):
            with st.spinner("Executing Resume Evaluation..."):
                start_time = time.time()
                results = []
                
                # Process resumes with progress tracking
                progress_bar = st.progress(0)
                for idx, file in enumerate(resume_files):
                    progress = (idx + 1) / len(resume_files)
                    progress_bar.progress(min(int(progress * 100), 100))
                    
                    resume_text = safe_read_file(file)
                    if resume_text:
                        analysis = analyze_resume(st.session_state.jd_text, resume_text)
                        results.append({
                            "file_name": file.name,
                            "score": analysis["total_score"],
                            "qualified": analysis["is_qualified"],
                            "details": analysis
                        })
                
                st.session_state.analysis_data = results
                st.success(f"Analysis completed in {time.time()-start_time:.1f} seconds")
        
        # Display Results
        if st.session_state.analysis_data:
            st.subheader("Executive Dashboard")
            df_data = []
            for result in st.session_state.analysis_data:
                df_data.append({
                    "Candidate": result["file_name"],
                    "Score": result["score"],
                    "Status": "Qualified" if result["qualified"] else "Review Needed",
                    "Recommendation": result["details"]["recommendation"]
                })
            
            summary_df = pd.DataFrame(df_data)
            st.dataframe(
                summary_df.sort_values("Score", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Detailed Candidate Reports
            st.subheader("In-Depth Candidate Analysis")
            for result in st.session_state.analysis_data:
                with st.expander(f"{result['file_name']} - {result['score']}/100"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric("Overall Score", f"{result['score']}/100")
                        st.metric("Hiring Recommendation", result["details"]["recommendation"])
                        
                        st.subheader("Competency Breakdown")
                        for section, data in result["details"]["section_details"].items():
                            st.write(f"**{section}**")
                            st.progress(data["score"]/100)
                            st.caption(f"{data['score']}/100 - {ESSENTIAL_SECTIONS[section]['analysis_prompt']}")
                    
                    with col2:
                        st.subheader("Evaluation")
                        for section, data in result["details"]["section_details"].items():
                            st.markdown(f"##### {section} Assessment")
                            
                            st.markdown("**Key Strengths**")
                            if data['matches']:
                                for match in data['matches']:
                                    st.markdown(f"- ✅ {match}")
                            else:
                                st.markdown("- No significant matches found")
                            
                            st.markdown("**Development Areas**")
                            if data['missing']:
                                for gap in data['missing']:
                                    st.markdown(f"- ❌ {gap}")
                            else:
                                st.markdown("- No critical deficiencies")
                            
                            st.markdown("**Expert Analysis**")
                            st.info(data['feedback'])
                            
                            st.write("---")

            st.subheader("Business Intelligence Export")
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary Report",
                data=csv_data,
                file_name="professional_assessment_report.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()