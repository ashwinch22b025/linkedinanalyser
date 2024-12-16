import streamlit as st
import os
from linkedin_api import Linkedin
import spacy
import en_core_web_sm
from spacy.matcher import Matcher
import PyPDF2
import openai

# Set up Streamlit interface
st.set_page_config(page_title='Profile and Resume Analyzer', page_icon=':clipboard:', layout='wide')
st.title('Profile and Resume Analyzer')

def authenticate_linkedin(client_id, client_secret):
    """
    Authenticate with the LinkedIn API using the provided client ID and secret.

    Args:
        client_id (str): LinkedIn API client ID.
        client_secret (str): LinkedIn API client secret.

    Returns:
        Linkedin: Authenticated LinkedIn API client instance, or None if authentication fails.
    """
    if client_id and client_secret:
        try:
            api = Linkedin(client_id, client_secret)
            return api
        except Exception as e:
            st.sidebar.error(f"LinkedIn API authentication failed: {e}")
            return None
    else:
        st.sidebar.error("Please set the LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET environment variables.")
        return None

def extract_text_from_pdf(file):
    """
    Extract text content from each page of a PDF file.

    Args:
        file (UploadedFile): PDF file uploaded by the user.

    Returns:
        str: Extracted text content of the PDF, or an error message if extraction fails.
    """
    text = ''
    try:
        pdf_reader = PyPDF2.PdfFileReader(file)
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def analyze_text(text):
    """
    Analyze text content using SpaCy to identify patterns and suggestions.

    Args:
        text (str): Text to be analyzed.

    Returns:
        tuple: SpaCy doc object and list of suggested improvements.
    """
    try:
        nlp = en_core_web_sm.load()
        doc = nlp(text)
        matcher = Matcher(nlp.vocab)
        pattern1 = [{'LOWER': 'improve'}, {'POS': 'ADP'}, {'POS': 'NOUN'}]
        pattern2 = [{'LOWER': 'add'}, {'POS': 'NOUN'}, {'POS': 'ADP'}, {'POS': 'NOUN'}]
        matcher.add('Improvement', [pattern1, pattern2])
        matches = matcher(doc)
        suggestions = [doc[start:end].text for match_id, start, end in matches]
        return doc, suggestions
    except Exception as e:
        st.error(f"Error analyzing text with SpaCy: {e}")
        return None, []

def generate_summary_and_recommendations(text):
    """
    Generate a summary and recommendations for improvement using OpenAI's GPT model.

    Args:
        text (str): Text to generate a summary and recommendations for.

    Returns:
        tuple: Summary and recommendations as strings, or an error message if generation fails.
    """
    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        summary = openai.Completion.create(
            engine='text-davinci-003',
            prompt=f"Please summarize the following text:\n{text}",
            temperature=0.5,
            max_tokens=100
        ).choices[0].text.strip()
        
        recommendations = openai.Completion.create(
            engine='text-davinci-003',
            prompt=f"Please provide recommendations for improvement based on the following text:\n{text}",
            temperature=0.5,
            max_tokens=200
        ).choices[0].text.strip()
        
        return summary, recommendations
    except Exception as e:
        st.error(f"Error generating summary and recommendations with OpenAI: {e}")
        return "Summary not available.", "Recommendations not available."

# Get LinkedIn API authentication from environment variables
client_id = os.getenv('LINKEDIN_CLIENT_ID')
client_secret = os.getenv('LINKEDIN_CLIENT_SECRET')

# Authenticate with LinkedIn API
api = authenticate_linkedin(client_id, client_secret)

# Get profile URL from user
st.sidebar.title('Profile URL')
profile_url = st.sidebar.text_input('Enter your LinkedIn profile URL')

if profile_url and api:
    try:
        # Extract profile data using LinkedIn API
        profile = api.get_profile(profile_url=profile_url)
        
        # Analyze profile using SpaCy
        doc_profile, suggestions_profile = analyze_text(profile.get('summary', ''))
        
        st.write('**Summary:**')
        st.write(profile.get('summary', 'No summary found.'))
        
        if suggestions_profile:
            st.write('**Suggestions:**')
            for suggestion in suggestions_profile:
                st.write(suggestion)
        else:
            st.write('**No suggestions. Your profile looks great!**')
    except Exception as e:
        st.error(f"Error fetching LinkedIn profile: {e}")

