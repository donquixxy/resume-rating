import os
import re
from io import BytesIO
from typing import BinaryIO, List

import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

placeholder = (
    "- Proficiency in Python and experience with backend development at least 2 years\n"
    "- Experience with RESTful APIs and microservices architecture\n"
)

help = (
    "- Avoid using generic terms such as : Detail-Oriented, Proactive, Motivated.\n"
    "- Instead, be specific and measurable. Mention tools, technologies, years of experience, project types, or actual responsibilities."
)

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

def summarize_chunks(chunks: list[str], requirements: str) -> str:
    model = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash-001"
    )

    joined_chunks = "\n".join(chunks)

    prompt = f"""
    You are a hiring assistant. Below are the job requirements and the candidate's resume content.

    ## Job Requirements:
    {requirements}

    ## Candidate Resume:
    {joined_chunks}

    Evaluate how well the candidate fits the job requirements. Also, dont return the candidate name

    Return the following:
    1. **Summary**: A short overview of the candidate’s fit.
    2. **Matched Skills**: List of job requirements that are satisfied.
    3. **Missing Skills**: List of key requirements that are not found.
    4. **Rating**: Score from 1 to 10 (10 = perfect match, 1 = not a match).
    5. **Final Verdict**: Suitable or Not Suitable

    Respond in a structured and professional tone.
    """

    response = model.invoke(prompt)
    return response.content.strip()

def generate_chunk(raw: str):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=80,
        chunk_overlap=30,
        encoding_name="cl100k_base"
    )

    r = text_splitter.split_text(raw)
    return r


def read_file_pdf(file: BinaryIO) -> list[str]:
    c = PdfReader(file)
    raw_text = ""
    for page in c.pages:
        raw_text += page.extract_text()

    trimmed_text = raw_text.replace("\t", " ")
    trimmed_text = re.sub(r"\s+", " ", trimmed_text)

    ### Read content in chunks
    result = generate_chunk(trimmed_text)

    return result


def read_file_words(file: BinaryIO) -> list[str]:
    doc = Document(BytesIO(file.read()))
    raw_text = ""
    for x in doc.paragraphs:
        raw_text += x.text

    ### Trim Text to cleaner, readable format
    trimmed_text = raw_text.replace("\t", " ")
    trimmed_text = re.sub(r"\s+", " ", trimmed_text)

    ### Read content in chunks
    result = generate_chunk(trimmed_text)
    return result



def main():
    ### Create streamlit header
    st.subheader("Rate your candidate Resume, Check wether they are matches your job 👀")

    text_input = st.text_area(label="Input job qualifications",  placeholder=placeholder, help=help)

    if text_input:
        if len(text_input) <  200:
            st.error("Your qualifications is too short. The more data you provide, the accurate our AI to check.")

    uploaded_file = st.file_uploader(label="Input resume ( pdf or docx)")

    if uploaded_file and text_input:
        ### Read value as byte
        file_name = uploaded_file.name

        file_type = file_name.split(".")[-1]
        result: list[str] = []
        match file_type:
            case "pdf":
               result =  read_file_pdf(uploaded_file)
            case ("docx"|"doc"):
               result =  read_file_words(uploaded_file)
            case _:
                st.error("Unsupported file type. Expected pdf or docx")

        if result:
                with st.spinner("Generating summary . . ."):
                    summary = summarize_chunks(chunks=result, requirements=text_input)
                    st.markdown("### Summary")
                    st.write(summary)


if __name__ == "__main__":
    main()
