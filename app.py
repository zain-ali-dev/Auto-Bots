import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.metrics import accuracy_score
import hashlib
import os
import time

# Function to create a hash for the URL
def get_url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

# Function to check if the content is cached
def is_cached(url):
    url_hash = get_url_hash(url)
    return os.path.exists(f'cache/{url_hash}.txt')

# Function to save content to cache
def save_to_cache(url, content):
    url_hash = get_url_hash(url)
    with open(f'cache/{url_hash}.txt', 'w', encoding='utf-8') as file:
        file.write(content)

# Function to load content from cache
def load_from_cache(url):
    url_hash = get_url_hash(url)
    with open(f'cache/{url_hash}.txt', 'r', encoding='utf-8') as file:
        return file.read()

# Function to fetch the content of the web page
def fetch_page_content(url, update_cache=False):
    if is_cached(url) and not update_cache:
        st.info("Loading data from cache...")
        st.session_state.is_cache_used = True
        return load_from_cache(url)
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        save_to_cache(url, content)
        st.session_state.is_cache_used = False
        return content
    else:
        st.error(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

# Function to extract text from the HTML content
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    page_text = soup.get_text(separator=' ', strip=True)
    return page_text

# Function to train a simple QA model using transformers pipeline
def train_model(data):
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

# Function to evaluate the model
def evaluate_model(model, data, questions, answers):
    predictions = [model(question=question, context=data)['answer'] for question in questions]
    accuracy = accuracy_score(answers, predictions)
    return accuracy

# Create cache directory if it doesn't exist
if not os.path.exists('cache'):
    os.makedirs('cache')

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'is_cache_used' not in st.session_state:
    st.session_state.is_cache_used = False

# Streamlit app
st.title("Web Scraping and QA Model Training")

# Step 1: Input URL
st.header("Step 1: Input URL")
url = st.text_input("Enter the URL of the site you want to scrape:")

fetch_data = st.button("Fetch Data")
if fetch_data:
    progress_bar = st.progress(0)
    with st.spinner("Fetching data..."):
        html_content = fetch_page_content(url)
        progress_bar.progress(25)
        if html_content:
            st.session_state.data = extract_text_from_html(html_content)
            progress_bar.progress(50)
            st.success("Data fetched successfully!")
            progress_bar.progress(100)
        else:
            progress_bar.progress(0)

# Conditionally show the "Update Data" button
if st.session_state.is_cache_used:
    update_data = st.button("Update Data")
    if update_data:
        progress_bar = st.progress(0)
        with st.spinner("Fetching data..."):
            html_content = fetch_page_content(url, update_cache=True)
            progress_bar.progress(25)
            if html_content:
                st.session_state.data = extract_text_from_html(html_content)
                progress_bar.progress(50)
                st.success("Data updated successfully!")
                progress_bar.progress(100)
            else:
                progress_bar.progress(0)

# Display extracted data if available
if st.session_state.data:
    st.text_area("Extracted Data", st.session_state.data, height=200)

    # Step 2: Train the Model
    st.header("Step 2: Train the Model")
    if st.button("Train Model"):
        progress_bar = st.progress(0)
        with st.spinner("Training model..."):
            st.session_state.model = train_model(st.session_state.data)
            progress_bar.progress(50)
            st.success("Model trained successfully!")
            progress_bar.progress(100)

# Display QA model interface if model is trained
if st.session_state.model:
    # Step 3: Ask Questions
    st.header("Step 3: Ask Questions")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        progress_bar = st.progress(0)
        with st.spinner("Getting answer..."):
            answer = st.session_state.model(question=question, context=st.session_state.data)['answer']
            progress_bar.progress(50)
            st.write(f"Answer: {answer}")
            progress_bar.progress(100)

    # (Optional) Step 4: Evaluate Model
    st.header("Step 4: Evaluate Model")
    if st.button("Evaluate Model"):
        progress_bar = st.progress(0)
        with st.spinner("Evaluating model..."):
            # Dummy data for evaluation (replace with real data)
            questions = ["What is the title?", "What is the main topic?"]
            answers = ["Example Title", "Example Topic"]
            accuracy = evaluate_model(st.session_state.model, st.session_state.data, questions, answers)
            progress_bar.progress(50)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
            progress_bar.progress(100)
