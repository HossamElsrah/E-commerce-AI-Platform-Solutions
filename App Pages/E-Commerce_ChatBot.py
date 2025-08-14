import os
import streamlit as st
import subprocess
from threading import Thread
import time
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from pyngrok import ngrok, conf
from kaggle_secrets import UserSecretsClient

# --- 1. API Keys (Must be replaced) ---
# WARNING: In a production environment, it is recommended to use Kaggle Secrets.
# IMPORTANT: Replace the following values with your actual API keys.
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- 2. ngrok Setup ---
# The ngrok auth token is hardcoded here as requested.
# Please remember that for security best practices, it is recommended to use
# Kaggle Secrets.
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# --- 3. Write the Streamlit app file ---
# This part of the code creates an app.py file containing the complete RAG application code.
app_py_content = """
import streamlit as st
import os
import torch
import transformers
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereRerank
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever

# IMPORTANT: Replace these values with your actual API keys.
HUGGINGFACE_TOKEN = "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"
COHERE_API_KEY = "BFRQTKUXpMbErJChHDGaiOyMQ6RjZ9VtRjCA580G"

@st.cache_resource
def get_rag_chain():
    \"\"\"
    Loads all components and assembles the RAG chain.
    This function runs only once due to @st.cache_resource.
    \"\"\"
    # Configure 4-bit quantization for efficient memory usage
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        device_map = "auto"
    else:
        st.warning("CUDA is not available. Running the model on CPU may be very slow.")
        quantization_config = None
        device_map = None

    # Load the LLM Model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
        token=HUGGINGFACE_TOKEN
    )
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    
    # Create the LLM pipeline
    pipeline = HuggingFacePipeline(
        pipeline=transformers.pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    )

    # Load the FAISS Index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index_path = "faiss_index"
    
    try:
        faiss_index = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index. Please ensure the 'faiss_index' directory exists and is correctly populated: {e}")
        st.stop()

    # Create the Reranker
    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=5)
    
    # Create the Retriever with Re-ranking
    base_retriever = faiss_index.as_retriever(search_kwargs={"k": 20})
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    # Create the Memory and Prompt
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
        llm=pipeline,
        retriever=compressed_retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
        }
    )
    
    return qa_chain

# --- Streamlit App UI ---
rag_chain = get_rag_chain()
st.set_page_config(page_title="E-Commerce Assistant", page_icon="🛍️")
st.title("E-Commerce Assistant")

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
            # Invoke the RAG chain
            response = rag_chain.invoke({"question": prompt})
            answer_text = response.get('answer', '')
            
            # --- Robust answer cleaning logic to only return the answer ---
            # This logic mimics the successful interactive code to ensure only the final answer is shown.
            cleaned_answer = ""
            # The model's output might contain the whole prompt, we need to extract the part after 'Answer:'
            if "Answer:" in answer_text:
                # Find the last occurrence of "Answer:" to handle cases where it might appear in chat history
                last_answer_index = answer_text.rfind("Answer:")
                cleaned_answer = answer_text[last_answer_index + len("Answer:"):].strip()
            else:
                # As a fallback, simply take the entire text if the label is missing
                cleaned_answer = answer_text.strip()
            
            st.markdown(cleaned_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": cleaned_answer})
"""
with open("app.py", "w") as f:
    f.write(app_py_content)

# --- 4. Run the Streamlit app with ngrok ---
# This part runs the Streamlit app and opens a public URL for it using ngrok
def run_streamlit_app():
    # Install pyngrok
    os.system("pip install pyngrok --quiet")

    # Start streamlit app in background
    proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

    # Wait for the Streamlit app to start on port 8501
    time.sleep(10)
    
    # Open ngrok tunnel to the Streamlit app
    try:
        public_url = ngrok.connect(8501).public_url
        print(f"Your Streamlit app is running at: {public_url}")
    except Exception as e:
        print(f"Failed to start ngrok tunnel: {e}")
        ngrok.kill()
        proc.kill()
        raise

run_streamlit_app()