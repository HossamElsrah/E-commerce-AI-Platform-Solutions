# E-commerce AI Recommendation and Analytics Platform

This project is an integrated, multi-app solution designed to empower e-commerce businesses by turning raw data into valuable, actionable insights. Leveraging the power of a large language model (LLM) and Retrieval-Augmented Generation (RAG), the platform provides two main tools to enhance business strategy and customer experience.

-----

## 🚀 Key Features

  * **Analytics Chatbot:** A powerful tool that allows users to upload a CSV file of their e-commerce data. Users can ask natural language questions about their sales, customer behavior, and product performance. The chatbot then provides data-driven insights and strategic recommendations.
  * **RAG Assistant:** A dedicated knowledge base assistant. This application uses Retrieval-Augmented Generation (RAG) to search through internal documentation or product information and provide quick, contextual, and accurate answers to specific queries.
  * **High-Performance Model:** The application utilizes the **`google/flan-t5-xl`** model to ensure high-quality and detailed responses, enhancing the accuracy of both the analytics insights and the RAG-based information retrieval.

-----

## 🧠 Approach and Methodology

Our methodology for developing this platform was focused on building a robust, data-driven system.

1.  **Data Analysis:** We collected and preprocessed e-commerce datasets, including user preferences, product information, and transaction history. This data was then used by the Analytics Chatbot to extract features and provide meaningful insights.
2.  **RAG Integration:** The RAG Assistant combines the power of the `flan-t5-xl` model with a vectorized knowledge base (FAISS index). This allows the system to retrieve relevant information from documents before generating a response, ensuring the answers are accurate and contextually relevant.
3.  **Model Deployment:** The **`flan-t5-xl`** LLM is the core of the system. It generates personalized recommendations, formats data query results into conversational insights, and provides detailed strategic advice.

-----

## 🛠️ Installation and Setup

To run this application, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-project.git
    cd your-project
    ```
2.  **Install Dependencies:** Make sure you have Python installed. We recommend using a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file would include `streamlit`, `pandas`, `transformers`, `torch`, `langchain`, `langchain-cohere`, `faiss-cpu`, `pyngrok`)*
3.  **Set Up Ngrok:** The application uses Ngrok to create a public URL for the Streamlit app. You will need to obtain an auth token from the Ngrok website.
    ```python
    # In the main app file (app.py), replace the placeholder with your token
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN" 
    ```
4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

-----

## 🚧 Challenges and Solutions

Developing this integrated system presented several challenges common with LLM-based projects:

  * **Out of Memory (OOM) Issues:** The large size of the `flan-t5-xl` model and the e-commerce datasets led to frequent memory errors.
      * **Solution:** We addressed this by implementing **4-bit quantization** using `BitsAndBytesConfig` and setting `device_map="auto"` when loading the model. This significantly reduced memory usage, allowing the application to run effectively.
  * **Library Compatibility:** We encountered issues with certain libraries, such as `pandas` agents, which were not fully compatible with the specific LLM versions used.
      * **Solution:** This was resolved by designing a custom agent with a fixed set of pandas expressions and a prompt-based approach to select the most relevant one, bypassing the compatibility issues.
  * **Version Conflicts:** Differences between library versions caused integration errors.
      * **Solution:** We carefully managed dependencies and ensured all libraries were at compatible versions for a seamless integration.

-----

## ✅ Achievements and Conclusion

Despite the technical hurdles, the project successfully demonstrated the powerful capabilities of combining a large language model with RAG for e-commerce.

  * **Enhanced Decision-Making:** The platform provides business owners with clear, data-driven insights and strategic recommendations that would otherwise be difficult to obtain manually.
  * **Improved User Experience:** By providing a natural language interface for data analytics and a reliable knowledge base, the system enhances the way businesses interact with their own information.
  * **Demonstrated Potential:** This project serves as a strong proof of concept for leveraging AI to streamline operations and provide a competitive edge in the fast-paced e-commerce landscape.

This project is a testament to the potential of AI to revolutionize business intelligence and decision-making. We invite you to explore the platform and see its capabilities firsthand.
