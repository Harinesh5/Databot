import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
try:
    llm = GoogleGemini(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")
    logger.info("Initialized GoogleGemini with model: gemini-2.5-flash")
except Exception as e:
    st.error(f"Error initializing Gemini model: {str(e)}")
    st.info("Please ensure your API key is valid and the model 'gemini-2.5-flash' is available in your region. Run: curl 'https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY' to check available models.")
    logger.error(f"Gemini initialization failed: {str(e)}")
    st.stop()

# Initialize Gemini model for general search
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("Initialized Gemini 2.5 Flash for general search")
except Exception as e:
    st.error(f"Error initializing Gemini search model: {str(e)}")
    logger.error(f"Gemini search initialization failed: {str(e)}")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Dataset Analysis & General Knowledge Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history and dataset
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "smart_df" not in st.session_state:
    st.session_state.smart_df = None

# Supported file formats
file_formats = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xls": pd.read_excel,
    "json": pd.read_json
}

# Function to load dataset
@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        if ext in file_formats:
            logger.info(f"Loading dataset: {uploaded_file.name}")
            return file_formats[ext](uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
            logger.error(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logger.error(f"Error loading file: {str(e)}")
        return None

# Function to process query with PandasAI
def process_query(query, smart_df):
    try:
        logger.info(f"Processing dataset query: {query}")
        response = smart_df.chat(query)
        logger.info(f"Query response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing dataset query: {str(e)}")
        return f"Error processing dataset query: {str(e)}"

# Function to handle general knowledge questions using Gemini
def search_general_knowledge(query):
    try:
        logger.info(f"Processing general knowledge query: {query}")
        response = gemini_model.generate_content(query)
        logger.info(f"General knowledge response received")
        return response.text
    except Exception as e:
        logger.error(f"Error processing general knowledge query: {str(e)}")
        return f"Error processing general knowledge query: {str(e)}"

# Streamlit UI
st.title("JUJU BOT")
st.caption("Upload a dataset to analyze it or ask general knowledge questions using Gemini 2.5 Flash!")

# Dataset section
st.subheader("Dataset Analysis")
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

# Load dataset if uploaded
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.df = df
        try:
            st.session_state.smart_df = SmartDataframe(df, config={"llm": llm, "verbose": True})
            st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            logger.info("SmartDataframe initialized successfully")
        except Exception as e:
            st.error(f"Error initializing SmartDataframe: {str(e)}")
            logger.error(f"Error initializing SmartDataframe: {str(e)}")
            st.session_state.smart_df = None

# Chat interface below dataset preview
st.subheader("Chat Interface")
query_type = st.radio("Select Query Type:", ("Dataset Query", "General Query"), horizontal=True)
user_input = st.chat_input("Ask a question about your dataset or general knowledge...")

# Display chat history
st.markdown("### Chat History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Process user query
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "MUFFIN", "content": user_input})
    
    # Process based on query type
    if query_type == "Dataset Query" and st.session_state.smart_df is not None:
        response = process_query(user_input, st.session_state.smart_df)
    elif query_type == "Dataset Query" and st.session_state.smart_df is None:
        response = "Please upload a dataset before asking dataset-related questions."
    else:  # General Knowledge
        response = search_general_knowledge(user_input)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "BLUEYOUS", "content": response})
    st.rerun()

# Sidebar for clearing chat history
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()