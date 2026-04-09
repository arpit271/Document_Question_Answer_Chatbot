# Document Question Answer Chatbot 🤖

Chat seamlessly with multiple PDFs using **LangChain**, **Azure OpenAI**, and **FAISS Vector DB** via **Streamlit**. Get instant, accurate answers from your uploaded PDFs. 📚💬

---

## 📝 Description
The Document Question Answer Chatbot is a **Streamlit-based web application** that allows users to upload multiple PDF documents, extract text from them, and create a chatbot using this content. Users can then ask questions in natural language and receive answers based on the PDF content using Azure OpenAI embeddings and chat models.

---

## 🎯 How It Works
1. **PDF Loading**: Reads multiple PDF documents and extracts text from each page.  
2. **Text Chunking**: Divides extracted text into smaller overlapping chunks for effective processing.  
3. **Embeddings**: Converts text chunks into vector embeddings using Azure OpenAI.  
4. **Similarity Matching**: Compares user questions with embeddings to find the most relevant chunks.  
5. **Response Generation**: Selected chunks are passed to an Azure OpenAI chat model to generate answers.

---

## 🌟 Key Features
- **Adaptive Chunking**: Sliding window chunking balances context and performance.  
- **Multi-Document QA**: Supports queries across multiple PDFs simultaneously.  
- **File Compatibility**: Handles PDF files efficiently.  
- **Azure OpenAI Support**: Works with GPT-4o-mini, GPT-3.5-turbo, or other Azure deployments.  

---

## ▶️ Installation
1. Clone the repository:

```bash
git clone https://github.com/arpit271/Document_Question_Answer_Chatbot.git
cd Document_Question_Answer_Chatbot
```


2. Install required Python packages:

```bash
pip install -r requirements.txt
```


3. Create a `.env` file in the project root with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key
```


4. Run the Streamlit app:

```bash
streamlit run app.py
```


## 💡 Usage
1. Upload multiple PDF files using the sidebar.  
2. Click **Submit & Process** to extract text and generate embeddings.  
3. Ask questions in natural language using the input field.  
4. The chatbot provides responses based on the PDF content.  

---

## 🌟 Requirements
- **streamlit**  
- **python-dotenv**  
- **langchain**  
- **PyPDF2**  
- **faiss-cpu**  
- **openai**

