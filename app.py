import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("📄 Resume Keyword Matcher")

st.write("Compare your resume with a job description and get match percentage.")

# Inputs
resume = st.text_area("Paste your Resume Here")
job_desc = st.text_area("Paste Job Description Here")

# Button
if st.button("Check Match"):
    if resume and job_desc:
        # Convert text into vectors
        text = [resume, job_desc]
        cv = CountVectorizer().fit_transform(text)
        similarity = cosine_similarity(cv)[0][1]

        # Show match %
        st.success(f"Match Percentage: {round(similarity * 100, 2)}%")

        # Find common keywords
        words_resume = set(resume.lower().split())
        words_job = set(job_desc.lower().split())
        common_words = words_resume.intersection(words_job)

        st.subheader("Common Keywords")
        st.write(common_words)

    else:
        st.error("Please enter both Resume and Job Description")
