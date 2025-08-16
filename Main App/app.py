import os
import subprocess
import time
import requests
import sys
from pyngrok import ngrok, conf
import shutil

# Set your ngrok auth token.
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# --- 1. Define the content for each app file ---

# Main app file (Home page)
main_app_content = """
import streamlit as st

st.set_page_config(page_title="E-commerce Solutions", page_icon="🛍️", layout="wide")
st.title("🛍️ Integrated E-commerce Solutions Platform")
st.write(\"\"\"
Welcome to the Integrated E-commerce Solutions Platform. This platform is specifically designed to help e-commerce store owners transform their data into valuable, actionable insights. By leveraging advanced AI tools, we provide you with the ability to understand customer behavior, analyze product performance, and make data-driven strategic and marketing decisions.

Our platform consists of two main applications:

1.  **Analytics Chatbot:**
    * Allows you to upload your sales data via CSV files.
    * Answers your questions in natural language about your store's performance.
    * Provides instant analytics on total sales, best-selling products, customer behavior, and more.
    * Helps you identify strengths and weaknesses in your business.

2.  **RAG Assistant:**
    * Acts as a knowledge base to help you get accurate and detailed information.
    * Uses Retrieval Augmented Generation (RAG) technology to search your documents and provide quick, relevant answers.
    * Ideal for getting answers about company policies, product details, or any other information you need.

We believe that understanding data is the key to success in the world of e-commerce. Use the sidebar to navigate between the applications and explore the analytical power of our platform.
\"\"\")
"""

# First page: Analytics Chatbot App
analytics_app_content = """
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import BitsAndBytesConfig

# Define the model to use. Now using the larger 't5-xl' model.
MODEL_NAME = "google/flan-t5-xl"

# --- Centralized LLM and Agent Logic ---
@st.cache_resource
def get_llm_and_agent_components():
    \"\"\"
    Loads LLM and defines the agent's expressions and prompts.
    This function is cached to ensure the model is loaded only once.
    \"\"\"
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load the XL model with quantization for memory efficiency.
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )
            device_map = "auto"
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                device_map=device_map,
                quantization_config=quantization_config,
            )
            llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=512)
        else:
            st.warning("CUDA is not available. Running the model on CPU may be very slow.")
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=512, device=-1)

        llm = HuggingFacePipeline(pipeline=llm_pipe)

        # --- IMPORTANT: These expressions are specific to your dataset columns. ---
        # A dictionary mapping a keyword to the corresponding pandas expression.
        PANDAS_EXPRESSIONS = {
            # General order information
            "total_orders": "len(data)",
            "order_statuses_count": "data['order_status'].value_counts()",
            "unique_product_categories_count": "data['category'].nunique()",
            
            # Financial insights
            "total_sales_value": "data['price'].sum()",
            "total_freight_value": "data['freight'].sum()",
            "total_payment_value": "data['payment_value'].sum()",
            "avg_payment_value": "data['payment_value'].mean()",
            "avg_freight_value": "data['freight'].mean()",
            "avg_price_per_product": "data['price'].mean()",
            "most_expensive_product_price": "data['price'].max()",
            "least_expensive_product_price": "data['price'].min()",
            "median_price": "data['price'].median()", 
            "std_dev_price": "data['price'].std()", 
            
            # Product and category insights
            "count_orders_per_category": "data.groupby('category')['order_status'].count()",
            "avg_price_per_category": "data.groupby('category')['price'].mean()",
            "top_3_categories_by_sales": "data.groupby('category')['price'].sum().nlargest(3)",
            "top_5_categories_by_sales": "data.groupby('category')['price'].sum().nlargest(5)",
            "bottom_5_categories_by_sales": "data.groupby('category')['price'].sum().nsmallest(5)",
            "most_popular_category": "data['category'].mode()[0]",
            
            # Payment and review analysis
            "reviews_per_category": "data.groupby('category')['review'].count()",
            "most_common_payment_type": "data['payment_method'].mode()[0]",
            "avg_review_score": "data['review'].mean()",
            "reviews_by_score": "data['review'].value_counts().sort_index()",
            "reviews_per_state": "data.groupby('customer_state')['review'].mean()",
            
            # Geographic data
            "city_with_most_orders": "data['customer_city'].mode()[0]",
            "state_with_most_orders": "data['customer_state'].mode()[0]",
            "top_5_cities_by_orders": "data['customer_city'].value_counts().nlargest(5)",
            "top_5_states_by_orders": "data['customer_state'].value_counts().nlargest(5)",
            "top_3_states_by_sales": "data.groupby('customer_state')['price'].sum().nlargest(3)",
            "sales_per_state": "data.groupby('customer_state')['price'].sum()", 
            "orders_by_state_and_city": "data.groupby(['customer_state', 'customer_city'])['order_status'].count().sort_values(ascending=False)",
        }
        
        RECOMMENDATION_PROMPT_TEMPLATE = \"\"\"
        You are an expert e-commerce marketing consultant. Your task is to provide a detailed, actionable, and comprehensive recommendation to a business owner based on a data-driven insight.

        Insight from data analyst: {insight}

        Detailed Recommendation:
        \"\"\"

        INSIGHT_FORMATTING_PROMPT = \"\"\"
        You are an expert data analyst. The user asked a question and you have the result of a data query.
        Please format the raw result into a clear, professional, and conversational sentence.
        Do not just print the numbers. Explain what they mean.
        User Query: {query}
        Raw Result: {raw_result}
        Formatted Answer:
        \"\"\"

        return llm, PANDAS_EXPRESSIONS, RECOMMENDATION_PROMPT_TEMPLATE, INSIGHT_FORMATTING_PROMPT
    except Exception as e:
        st.error(f"Error loading model: {e}. Please try again or use a smaller model.")
        return None, None, None, None

def run_query_with_llm(query, df, llm, expressions, insight_prompt, recommendation_prompt_template):
    \"\"\"
    Identifies the best pandas expression, executes it, and generates insight and recommendation.
    \"\"\"
    keyword_prompt = f\"\"\"
    You are an expert data analyst. You are given a pandas DataFrame named 'data'.
    Your task is to identify which of the following keywords best answers the user's query:
    Keywords: {list(expressions.keys())}
    Please provide only the single keyword that is the best match. Do not provide any other text or explanation.
    Query: {query}
    Response:
    \"\"\"
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
st.set_page_config(page_title="Analytics Chatbot", page_icon="📊", layout="wide")
st.title("📊 Analytics Chatbot")
st.write("Upload a CSV file and ask questions about your data.")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data" not in st.session_state:
    st.session_state.data = None

# --- File Upload Section ---
st.header("Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
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
                if llm:
                    insight, recommendation = run_query_with_llm(
                        prompt, 
                        st.session_state.data, 
                        llm, 
                        PANDAS_EXPRESSIONS, 
                        INSIGHT_FORMATTING_PROMPT, 
                        RECOMMENDATION_PROMPT_TEMPLATE
                    )
                    
                    full_response = f\"\"\"**Insight:**
{insight}

**Recommendation:**
{recommendation}\"\"\"
                    st.markdown(full_response)
                else:
                    st.error("Failed to load the LLM model. Please check the model name or available resources.")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a CSV file to begin.")
"""

# Second page: RAG Assistant App
rag_app_content = """
import streamlit as st
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.docstore.document import Document
from transformers import BitsAndBytesConfig

# IMPORTANT: The provided API keys are not strictly needed as we are using a local LLM, but they might be
# used for other components like Cohere Rerank.
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# Define the model to use. Now using the larger 't5-xl' model.
MODEL_NAME = "google/flan-t5-xl"

# --- Centralized LLM and RAG Logic ---
@st.cache_resource
def get_rag_chain_components():
    \"\"\"
    Loads all components and assembles the RAG chain.
    This function runs only once due to @st.cache_resource.
    \"\"\"
    try:
        # Load the LLM Model and tokenizer (using the same model as the other app)
        llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )
            device_map = "auto"
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                device_map=device_map,
                quantization_config=quantization_config,
            )
            # Removed the device argument from the pipeline
            llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=512)
        else:
            st.warning("CUDA is not available. Running the model on CPU may be very slow.")
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            llm_pipe = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=512, device=-1)
        
        llm_instance = HuggingFacePipeline(pipeline=llm_pipe)

        # Load the FAISS Index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index_path = "faiss_index"
        
        if not os.path.exists(faiss_index_path):
            st.warning("FAISS index not found. Creating a dummy index for demonstration.")
            docs = [Document(page_content="This is a document about a product.", metadata={"source": "dummy"})]
            faiss_index = FAISS.from_documents(docs, embeddings)
            faiss_index.save_local(faiss_index_path)
            # Re-load the index after creation
            faiss_index = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            faiss_index = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # Create the Reranker
        reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=5)
        
        # Create the Retriever with Re-ranking
        base_retriever = faiss_index.as_retriever(search_kwargs={"k": 20})
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        # Create the Memory and Prompt for a T5 model
        prompt_template = \"\"\"You are an e-commerce assistant. Use the following context and chat history to answer the question.
        If the answer is not in the provided documents, say "I don't know the answer based on the provided context."
        
        Chat History:
        {chat_history}
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:\"\"\"
        
        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True
        )

        # Assemble the full Conversational RAG Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_instance,
            retriever=compressed_retriever,
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
            }
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        st.warning("Please ensure all dependencies are installed and the FAISS index is available.")
        st.stop()


# --- Streamlit App UI ---
# We call the cached function here to retrieve the RAG chain.
rag_chain = get_rag_chain_components()
st.set_page_config(page_title="E-Commerce Assistant", page_icon="🛒")
st.title("🛒 E-Commerce Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"question": prompt})
            answer_text = response.get('answer', '')
            
            cleaned_answer = ""
            if "Answer:" in answer_text:
                last_answer_index = answer_text.rfind("Answer:")
                cleaned_answer = answer_text[last_answer_index + len("Answer:"):].strip()
            else:
                cleaned_answer = answer_text.strip()
            
            st.markdown(cleaned_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": cleaned_answer})
"""

# --- 2. Create the file structure ---
print("Creating file structure...")
# Remove the old 'pages' directory if it exists to ensure a clean start
if os.path.exists("pages"):
    shutil.rmtree("pages")

# Create the main app file
with open("app.py", "w") as f:
    f.write(main_app_content)

# Create the 'pages' directory
pages_dir = "pages"
os.makedirs(pages_dir, exist_ok=True)

# Create the first page file
with open(os.path.join(pages_dir, "1_Analytics_App.py"), "w") as f:
    f.write(analytics_app_content)

# Create the second page file
with open(os.path.join(pages_dir, "2_RAG_Assistant_App.py"), "w") as f:
    f.write(rag_app_content)

print("File structure created successfully.")

# --- 3. Run the Streamlit app with ngrok ---
def run_streamlit_app():
    # Kill all existing ngrok processes before starting a new one
    print("Killing any existing ngrok processes...")
    try:
        ngrok.kill()
    except Exception as e:
        print(f"No ngrok processes to kill or an error occurred: {e}")

    # Start streamlit app in background
    print("Starting Streamlit app...")
    proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

    # Wait for the Streamlit app to start
    print("Waiting for Streamlit app to start...")
    timeout = 180 
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8501")
            if response.status_code == 200:
                print("Streamlit app is running!")
                break
        except requests.exceptions.ConnectionError:
            print("Streamlit not ready yet. Retrying in 5 seconds...")
            time.sleep(5)
    else:
        print("Error: Streamlit app failed to start within the timeout period.")
        proc.kill()
        return

    # Open ngrok tunnel
    try:
        public_url = ngrok.connect(8501).public_url
        print(f"Your Streamlit app is running at: {public_url}")
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        ngrok.kill()
        proc.kill()
        raise

# Call the function to run the app
run_streamlit_app()
