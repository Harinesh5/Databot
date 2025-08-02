import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from dotenv import load_dotenv
import logging
import uuid

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
llm = GoogleGemini(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")
logger.info("Initialized GoogleGemini with model: gemini-2.5-flash")

# Initialize Gemini model for general search
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
logger.info("Initialized Gemini 2.5 Flash for general search")

# Streamlit page configuration
st.set_page_config(
    page_title="JUJU BOT",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history and datasets
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "datasets" not in st.session_state:
    st.session_state.datasets = {}  # Dictionary to store multiple datasets
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None

# Supported file formats
file_formats = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xls": pd.read_excel,
    "json": pd.read_json
}

# Function to load dataset
@st.cache_data(ttl="2h")
def load_data(uploaded_file, file_id):
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
        logger.info(f"Query response type: {type(response)}, value: {response}")
        
        # Convert response to DataFrame if possible
        if isinstance(response, pd.DataFrame):
            return response
        elif isinstance(response, (list, dict)):
            try:
                df_response = pd.DataFrame(response)
                logger.info("Converted response to DataFrame")
                return df_response
            except Exception as e:
                logger.warning(f"Failed to convert response to DataFrame: {str(e)}")
                return response
        else:
            logger.warning(f"Response is not a DataFrame: {type(response)}")
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
st.caption("Upload multiple datasets to analyze them or ask general knowledge questions using Gemini 2.5 Flash!")

# Dataset section
st.subheader("Dataset Analysis")
uploaded_files = st.file_uploader(
    "Upload your datasets (CSV, Excel, JSON)", 
    type=["csv", "xlsx", "xls", "json"], 
    accept_multiple_files=True
)

# Load datasets if uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = str(uuid.uuid4())  # Unique ID for each dataset
        if uploaded_file.name not in st.session_state.datasets:
            df = load_data(uploaded_file, file_id)
            if df is not None:
                st.session_state.datasets[uploaded_file.name] = {
                    "df": df,
                    "smart_df": SmartDataframe(df, config={"llm": llm, "verbose": True})
                }
                st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
                logger.info(f"SmartDataframe initialized for {uploaded_file.name}")
    
    # Set default active dataset if none selected
    if st.session_state.active_dataset is None and st.session_state.datasets:
        st.session_state.active_dataset = list(st.session_state.datasets.keys())[0]

# Display dataset previews in an expander
if st.session_state.datasets:
    with st.expander("Dataset Previews"):
        for dataset_name, data in st.session_state.datasets.items():
            st.subheader(f"Preview: {dataset_name}")
            st.dataframe(data["df"].head())

# Select active dataset
if st.session_state.datasets:
    st.session_state.active_dataset = st.selectbox(
        "Select Active Dataset",
        options=list(st.session_state.datasets.keys()),
        index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) if st.session_state.active_dataset in st.session_state.datasets else 0
    )

# Chat interface below dataset preview
st.subheader("Chat Interface")
query_type = st.radio("Select Query Type:", ("Dataset Query", "General Knowledge"), horizontal=True)

# Chat input
user_input = st.chat_input("Ask a question about your dataset or general knowledge...")

# Display chat history
st.markdown("### Chat History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "BLUEYOUS" and isinstance(msg["content"], pd.DataFrame):
            st.markdown(f"**{msg['role'].capitalize()}:**")
            st.dataframe(msg["content"])
        else:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Process user query
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "MUFFIN", "content": user_input})
    
    # Process based on query type
    if query_type == "Dataset Query" and st.session_state.active_dataset and st.session_state.datasets:
        # Process dataset query
        smart_df = st.session_state.datasets[st.session_state.active_dataset]["smart_df"]
        response = process_query(user_input, smart_df)
        
        # Display response in structured format
        if isinstance(response, pd.DataFrame):
            st.session_state.chat_history.append({"role": "BLUEYOUS", "content": response})
            st.subheader("Query Result (Table)")
            st.dataframe(response)
        else:
            response_text = response
            st.session_state.chat_history.append({"role": "BLUEYOUS", "content": response_text})
            st.subheader("Query Result")
            st.markdown(response_text)
            st.warning("Result is not a table. Displaying as text. If you expected a table, try rephrasing the query.")
    elif query_type == "Dataset Query" and (not st.session_state.active_dataset or not st.session_state.datasets):
        response_text = "Please upload a dataset and select it before asking dataset-related questions."
        st.session_state.chat_history.append({"role": "BLUEYOUS", "content": response_text})
    else:  # General Knowledge
        response_text = search_general_knowledge(user_input)
        st.session_state.chat_history.append({"role": "BLUEYOUS", "content": response_text})
    
    st.rerun()

# Sidebar for clearing chat history and removing datasets
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.session_state.datasets:
        st.subheader("Remove Datasets")
        dataset_to_remove = st.selectbox("Select Dataset to Remove", options=["None"] + list(st.session_state.datasets.keys()))
        if dataset_to_remove != "None" and st.button("Remove Selected Dataset"):
            del st.session_state.datasets[dataset_to_remove]
            if st.session_state.active_dataset == dataset_to_remove:
                st.session_state.active_dataset = list(st.session_state.datasets.keys())[0] if st.session_state.datasets else None
            st.rerun()