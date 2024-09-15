# 💡 Intelligent Chatbot-Powered Website-Based Question Answering System Using RAG and LLM

Welcome to the **Intelligent Chatbot-Powered Website-Based Question Answering System**! This project harnesses the power of **Large Language Models (LLMs)**, specifically OpenAI’s GPT-4, and **Retrieval-Augmented Generation (RAG)** to create an intelligent, context-aware chatbot. It is designed to provide accurate, insightful responses to user queries based on live web content. Whether you're developing a knowledge base, enhancing customer support, or simply offering web-based insights, this system is here to help!

## 🌟 Key Features

- ** Retrieval-Augmented Generation (RAG)**: Combines document retrieval with LLM-based response generation for more accurate, contextual answers.
- ** Website Content Integration**: Seamlessly load, split, and index web-based documents for easy search and retrieval.
- ** Chatbot Interface**: Provides a conversational interface for users to ask questions and get fact-based answers.
- ** Vector Store Creation**: Converts text chunks into embeddings using OpenAI’s state-of-the-art embeddings API and stores them in a vector database.
- ** Modular & Extensible**: Easily adaptable to new document sources or expanded functionality.
- ** Efficient & Scalable**: Handles large volumes of content and queries with optimized performance.

## 🚀 How It Works

1. **Load Web Content**: Documents are loaded from a specified URL using the `WebBaseLoader`. This allows the system to scrape and extract relevant web-based information.
2. **Text Splitting**: The extracted documents are split into manageable text chunks using the `RecursiveCharacterTextSplitter` for more efficient processing.
3. **Vector Store Creation**: Using OpenAI’s embeddings, text chunks are embedded into a vector store powered by Chroma, enabling fast similarity search and retrieval.
4. **RAG Chain Setup**: A RAG chain is created, which combines context-based retrieval with GPT-4 to generate answers to user queries.
5. **Question Answering**: Users ask questions via the chatbot, and the system retrieves relevant document chunks before generating an accurate, context-driven answer.

## 🛠️ Tech Stack

- **LangChain**: Framework for connecting LLMs with external tools.
- **OpenAI GPT-4**: Large language model for generating insightful and accurate answers.
- **Chroma**: Vector database for embedding and indexing document chunks.
- **RAG**: Combines document retrieval with LLM-based generation to enhance the accuracy of responses.

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo-name/chatbot-rag-llm.git
   cd chatbot-rag-llm

2. Install dependencies:

   ```bash
   pip install -r requirements.txt

3. Set Up your envirnment variables:
   ```bash
   export OPENAI_API_KEY=<your-openai-api-key>
   export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"


## 🎮 Usage

Once the environment is set up, you can generate answers to questions based on website content.
   ```bash
   docs = load_docs('https://example.com')
   answer = gen_answer('https://example.com', 'Your question here')
   print(answer)


