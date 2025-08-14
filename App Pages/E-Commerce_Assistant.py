# --- 2. Write the Streamlit app file ---
# The app logic is saved to a file named 'app.py'
app_py_content = '''
import streamlit as st
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import os
import torch

# IMPORTANT: The provided API keys are not strictly needed for this
# application, as we are using a free, local model (flan-t5-xl) and not a reranker.
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- LLM and Agent Logic ---
@st.cache_resource
def get_llm_and_agent_components():
    """Loads LLM and defines the agent's expressions and prompts."""
    # Load the specified larger model: flan-t5-xl
    # Switching from 'google/flan-t5-xl' to 'google/flan-t5-base' to save memory.
    llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    
    # Check for GPU and set device accordingly
    device = 0 if torch.cuda.is_available() else -1
    llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=512, device=device)
    llm = HuggingFacePipeline(pipeline=llm_pipe)

    # --- IMPORTANT: These expressions are specific to your dataset columns. ---
    # The agent's logic relies on these column names. If you upload a different CSV,
    # you may need to adjust these expressions to match your new column names.
    PANDAS_EXPRESSIONS = {
        "total_orders": "len(data)",
        "order_statuses_count": "data['order_status'].value_counts()",
        "unique_product_categories_count": "data['category'].nunique()",
        "total_sales_value": "data['price'].sum()",
        "avg_price_per_product": "data['price'].mean()",
        "count_orders_per_category": "data.groupby('category')['order_status'].count()",
        "avg_price_per_category": "data.groupby('category')['price'].mean()",
        "top_5_categories_by_sales": "data.groupby('category')['price'].sum().nlargest(5)",
        "most_common_payment_type": "data['payment_method'].mode()[0]",
        "avg_review_score": "data['review'].mean()",
        "top_5_states_by_orders": "data['customer_state'].value_counts().nlargest(5)",
        "sales_per_state": "data.groupby('customer_state')['price'].sum()",
    }
    
    RECOMMENDATION_PROMPT_TEMPLATE = """
    You are an expert e-commerce marketing consultant. Your task is to provide a detailed, actionable, and comprehensive recommendation to a business owner based on a data-driven insight.

    Insight from data analyst: {insight}

    Detailed Recommendation:
    """

    INSIGHT_FORMATTING_PROMPT = """
    You are an expert data analyst. The user asked a question and you have the result of a data query.
    Please format the raw result into a clear, professional, and conversational sentence.
    Do not just print the numbers. Explain what they mean.
    User Query: {query}
    Raw Result: {raw_result}
    Formatted Answer:
    """

    return llm, PANDAS_EXPRESSIONS, RECOMMENDATION_PROMPT_TEMPLATE, INSIGHT_FORMATTING_PROMPT

def run_query_with_llm(query, df, llm, expressions, insight_prompt, recommendation_prompt_template):
    """
    Identifies the best pandas expression, executes it, and generates insight and recommendation.
    """
    keyword_prompt = f"""
    You are an expert data analyst. You are given a pandas DataFrame named 'data'.
    Your task is to identify which of the following keywords best answers the user's query:
    Keywords: {list(expressions.keys())}
    Please provide only the single keyword that is the best match. Do not provide any other text or explanation.
    Query: {query}
    Response:
    """
    try:
        keyword = llm.invoke(keyword_prompt).strip()
        if keyword in expressions:
            expression_to_run = expressions[keyword]
            raw_result = eval(expression_to_run, {'data': df, 'pd': pd})
            
            insight_prompt_filled = insight_prompt.format(query=query, raw_result=raw_result)
            insight = llm.invoke(insight_prompt_filled).strip()

            recommendation_prompt_filled = recommendation_prompt_template.format(insight=insight)
            recommendation = llm.invoke(recommendation_prompt_filled).strip()
            
            return insight, recommendation
        else:
            return "Sorry, I can't find an appropriate analysis for this question.", ""
    except Exception as e:
        return f"An error occurred: {e}", ""

# --- Streamlit App UI ---
st.set_page_config(page_title="E-commerce Analytics Chatbot", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce Analytics Chatbot")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None

# --- File Upload Section ---
st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully! You can now start the chat below.")
        st.dataframe(st.session_state.data.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state.data = None

# --- Chat Interface ---
if st.session_state.data is not None:
    st.markdown("---")
    st.header("Start Chatting")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about your data?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                llm, PANDAS_EXPRESSIONS, RECOMMENDATION_PROMPT_TEMPLATE, INSIGHT_FORMATTING_PROMPT = get_llm_and_agent_components()
                insight, recommendation = run_query_with_llm(
                    prompt, 
                    st.session_state.data, 
                    llm, 
                    PANDAS_EXPRESSIONS, 
                    INSIGHT_FORMATTING_PROMPT, 
                    RECOMMENDATION_PROMPT_TEMPLATE
                )
                
                # --- This multi-line f-string is now correctly formatted ---
                full_response = f"""**Insight:**
{insight}

**Recommendation:**
{recommendation}"""
                st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a CSV file to begin.")
'''
with open("app.py", "w") as f:
    f.write(app_py_content)

# --- 3. Run the Streamlit app with ngrok ---
# This part runs the Streamlit app and opens a public URL for it using ngrok
import subprocess
import time
import requests
from pyngrok import ngrok, conf
import os

# Set your ngrok auth token.
NGROK_AUTH_TOKEN = "31CpEH0DnLmMSw7dD2KeI7wYvq4_3bf1GVWy4pAPVTqRvGoB9"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

def run_streamlit_app():
    # Start streamlit app in background
    # Note: Streamlit may take a bit longer to start with a larger model (flan-t5-xl)
    proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

    # Wait for the Streamlit app to start
    print("Waiting for Streamlit app to start...")
    # Add a counter for timeout
    timeout = 180 # Increased timeout to 3 minutes for a very large model
    start_time = time.time()
    
    # Check if the server is up and running before connecting ngrok
    while time.time() - start_time < timeout:
        try:
            # Try to connect to the Streamlit port
            response = requests.get("http://localhost:8501")
            if response.status_code == 200:
                print("Streamlit app is running!")
                break
        except requests.exceptions.ConnectionError:
            print("Streamlit not ready yet. Retrying in 5 seconds...")
            time.sleep(5)
    else:
        # If the loop completes without breaking, the app failed to start
        print("Error: Streamlit app failed to start within the timeout period.")
        proc.kill()
        return

    # Open ngrok tunnel
    try:
        # Kill all existing ngrok tunnels before creating a new one
        ngrok.kill()
        public_url = ngrok.connect(8501).public_url
        print(f"Your Streamlit app is running at: {public_url}")
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        ngrok.kill()
        proc.kill()
        raise

# Call the function to run the app
run_streamlit_app()