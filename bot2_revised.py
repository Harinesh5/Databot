import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
llm = GoogleGemini(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")

# Initialize Gemini model for general search
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Streamlit page configuration
st.set_page_config(
    page_title="JUJU BOT",
    page_icon="👫",
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
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        if ext in file_formats:
            return file_formats[ext](uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to generate a count plot
def generate_count_plot(df, x_col, hue_col):
    try:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=x_col, hue=hue_col)
        plt.title(f"Count of {x_col} per {hue_col}")
        plt.xlabel(x_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(plt)
        return plt.gcf()  # Return the figure for chat history
    except Exception as e:
        return f"Error generating count plot: {str(e)}"

# Function to parse graph query
def parse_graph_query(query, df):
    # Look for patterns like "graph/plot for Count of X per Y"
    pattern = r"(?:generate|plot|graph|show).*count\s+of\s+(\w+)\s+(?:per|by|with)\s+(\w+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match and len(match.groups()) == 2:
        x_col, hue_col = match.groups()
        if x_col in df.columns and hue_col in df.columns:
            return x_col, hue_col
    return None, None

# Function to process query with PandasAI
def process_query(query, smart_df):
    try:
        response = smart_df.chat(query)
        # Convert response to DataFrame if possible
        if isinstance(response, pd.DataFrame):
            return response
        elif isinstance(response, (list, dict)):
            try:
                df_response = pd.DataFrame(response)
                return df_response
            except Exception as e:
                return response
        else:
            return response
    except Exception as e:
        return f"Error processing dataset query: {str(e)}"

# Function to handle general knowledge questions using Gemini
def search_general_knowledge(query):
    try:
        response = gemini_model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error processing general query: {str(e)}"

# Streamlit UI
st.title("👫JUJU BOT")
st.caption("Upload multiple datasets to analyze them, generate graphs via queries")

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
        if uploaded_file.name not in st.session_state.datasets:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.datasets[uploaded_file.name] = {
                    "df": df,
                    "smart_df": SmartDataframe(df, config={"llm": llm, "verbose": True})
                }
                st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")
    
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
query_type = st.radio("Select Query Type:", ("Dataset Query", "General Query"), horizontal=True)

# Chat input
user_input = st.chat_input("Ask your question...")

# Display chat history
st.markdown("### Chat History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "BLUEYOUS(👧)" and isinstance(msg["content"], pd.DataFrame):
            st.markdown(f"**{msg['role'].capitalize()}:**")
            st.dataframe(msg["content"])
        elif msg["role"] == "BLUEYOUS(👧)" and isinstance(msg["content"], plt.Figure):
            st.markdown(f"**{msg['role'].capitalize()}:**")
            st.pyplot(msg["content"])
        else:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Process user query
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "MUFFIN(👦)", "content": user_input})
    
    # Process based on query type
    if query_type == "Dataset Query" and st.session_state.active_dataset and st.session_state.datasets:
        # Check if the query is requesting a graph
        df = st.session_state.datasets[st.session_state.active_dataset]["df"]
        x_col, hue_col = parse_graph_query(user_input, df)
        if x_col and hue_col:
            fig = generate_count_plot(df, x_col, hue_col)
            if isinstance(fig, plt.Figure):
                st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": fig})
                st.subheader("Graph Result")
                st.pyplot(fig)
            else:
                st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": fig})
                st.subheader("Graph Result")
                st.markdown(fig)
        else:
            # Process dataset query
            smart_df = st.session_state.datasets[st.session_state.active_dataset]["smart_df"]
            response = process_query(user_input, smart_df)
            
            # Display response in structured format
            if isinstance(response, pd.DataFrame):
                st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": response})
                st.subheader("Query Result (Table)")
                st.dataframe(response)
            else:
                response_text = response
                st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": response_text})
                st.subheader("Query Result")
                st.markdown(response_text)
                st.warning("Result is not a table. Displaying as text. If you expected a table, try rephrasing the query.")
    elif query_type == "Dataset Query" and (not st.session_state.active_dataset or not st.session_state.datasets):
        response_text = "Please upload a dataset and select it before asking dataset-related questions."
        st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": response_text})
    else:  # General Knowledge
        response_text = search_general_knowledge(user_input)
        st.session_state.chat_history.append({"role": "BLUEYOUS(👧)", "content": response_text})
    
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