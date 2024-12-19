import streamlit as st
import http.client
import urllib.parse
import json
import requests

# API Keys
OPENROUTER_API_KEY = "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"
OPEN_AI_KEY = ""

# Functions
def get_linkedin_user_details(linkedin_url):
    encoded_url = urllib.parse.quote(linkedin_url, safe='')
    conn = http.client.HTTPSConnection("fresh-linkedin-profile-data.p.rapidapi.com")
    headers = {
        'X-RapidAPI-Key': "f51b4efdc3mshe0f49f00da66748p1afa3ajsn7e4fd3128ea4",
        'X-RapidAPI-Host': "fresh-linkedin-profile-data.p.rapidapi.com"
    }
    conn.request("GET", f"/get-linkedin-profile?linkedin_url={encoded_url}&include_skills=false", headers=headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    return json.loads(data)

def get_linkedin_posts(linkedin_url):
    encoded_url = urllib.parse.quote(linkedin_url, safe='')
    conn = http.client.HTTPSConnection("fresh-linkedin-profile-data.p.rapidapi.com")
    headers = {
        'X-RapidAPI-Key': "f51b4efdc3mshe0f49f00da66748p1afa3ajsn7e4fd3128ea4",
        'X-RapidAPI-Host': "fresh-linkedin-profile-data.p.rapidapi.com"
    }
    conn.request("GET", f"/get-profile-posts?linkedin_url={encoded_url}&type=posts", headers=headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    return json.loads(data)

def generate_text(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "x-api-key": f"{OPEN_AI_KEY}",
        "Referer": "YOUR_SITE_URL",
        "X-Title": "YOUR_APP_NAME",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4-turbo",
        "max_tokens": 4000,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return response.json()

# Streamlit App
def main():
    st.title("LinkedIn Profile Analyzer")

    with st.form(key="linkedin_form"):
        linkedin_url = st.text_input("Enter the LinkedIn profile URL:")
        submit_button = st.form_submit_button(label="Analyze")

    if submit_button and linkedin_url:
        with st.spinner("Fetching user details..."):
            user_details = get_linkedin_user_details(linkedin_url)

        if 'data' in user_details:
            user_details_no_urls = {
                key: value for key, value in user_details['data'].items()
                if not isinstance(value, str) or 'http' not in value
            }
            st.subheader("User Details")
            st.json(user_details_no_urls)

            with st.spinner("Fetching user posts..."):
                posts_data = get_linkedin_posts(linkedin_url)
                posts = posts_data.get('data', [])

            if posts:
                st.subheader("Extracted Posts")
                for i, post in enumerate(posts, 1):
                    if 'text' in post:
                        st.markdown(f"**Post {i}:** {post['text']}")
                    else:
                        st.markdown(f"**Post {i}:** [No text available]")

                with st.spinner("Generating detailed analysis..."):
                    prompt = f"""
                    LinkedIn Profile Analysis

                    User Summary:
                    (Replace the example summary below with actual user details):
                    - Name: {user_details_no_urls.get('name', 'N/A')}
                    - Profile Summary: {user_details_no_urls.get('headline', 'N/A')}
                    - Profile URL: {linkedin_url}

                    Detailed Analysis Request:
                    1. Analyze the technical content of the user's posts.
                    2. Extract key phrases or important sentences that showcase expertise.
                    3. Assess engagement levels of the posts (likes, comments, shares).
                    4. Identify trends in discussed topics and alignment with industry trends.
                    5. Evaluate community impact and collaborative efforts.

                    Posts Analysis:
                    """
                    for i, post in enumerate(posts, 1):
                        try:
                            prompt += f"\nPost {i}: {post['text']}"
                        except KeyError:
                            prompt += f"\nPost {i}: [No text available]"

                    analysis_results = generate_text(prompt)

                if 'choices' in analysis_results:
                    st.subheader("Analysis Results")
                    for choice in analysis_results['choices']:
                        st.write(choice['message']['content'])
                else:
                    st.error("Failed to generate analysis results.")
            else:
                st.warning("No posts found or no text available in posts.")
        else:
            st.error("Failed to fetch user details. Please check the LinkedIn URL.")

if __name__ == "__main__":
    main()
