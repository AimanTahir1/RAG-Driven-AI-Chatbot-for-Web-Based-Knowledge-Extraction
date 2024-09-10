import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from secretkey import openapi_key

# Set environment variables for OpenAI API key and user agent
os.environ['OPENAI_API_KEY'] = openapi_key
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

def load_docs(url):
    """
    Load web documents from a specified URL.

    This function uses the WebBaseLoader to extract documents from the provided URL.

    Parameters:
    url (str): URL of the web page to scrape documents from.

    Returns:
    docs (list): A list of loaded documents.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def get_text_chunks(docs):
    """
    Split the loaded documents into smaller, manageable text chunks.

    This function uses the RecursiveCharacterTextSplitter to break the documents 
    into chunks of text with a specified overlap and size for easier processing.

    Parameters:
    docs (list): List of documents to be split into chunks.

    Returns:
    chunks (list): List of smaller text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vector_store(text_chunks):
    """
    Create a vector store and retriever from text chunks.

    This function generates embeddings for the text chunks and stores them in a 
    Chroma vector store, enabling efficient similarity search and retrieval.

    Parameters:
    text_chunks (list): List of text chunks to be embedded and stored.

    Returns:
    vectorstore (Chroma): The Chroma vector store containing the embeddings.
    retriever (Retriever): The retriever object for querying the vector store.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return vectorstore, retriever

def format_docs(docs):
    """
    Format documents into a single string of text.

    This function concatenates all the pages from the loaded documents into one 
    continuous string, ensuring the text can be processed by the RAG chain.

    Parameters:
    docs (list): List of documents to format.

    Returns:
    formatted_docs (str): A single formatted string containing the document contents.
    """
    return "\n".join([doc.page_content for doc in docs])

def get_rag_chain(retriever):
    """
    Set up a RAG (Retrieval-Augmented Generation) chain for answering questions.

    This function creates a RAG chain using a custom prompt template, 
    a retriever to supply context, and an LLM (GPT-4) to generate responses.

    Parameters:
    retriever (Retriever): The retriever used to query the vector store.

    Returns:
    rag_chain (Chain): A RAG chain configured with a custom prompt and LLM.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    template = """
    SYSTEM: You are a question-answer bot. 
    Be factual in your response.
    Respond to the following question: {question} only from 
    the below context: {context}. 
    If you don't know the answer, just say that you don't know.
    If the {question} is empty, say 'please enter the question.'
    """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    return rag_chain

def gen_answer(url, question):
    """
    Generate an answer to a given question based on content from a specified URL.

    This is the main function that combines all steps: loading documents from a URL, 
    splitting the content into chunks, creating a vector store, and using a RAG chain 
    to generate an answer to a question.

    Parameters:
    url (str): The URL to scrape documents from.
    question (str): The user's question that the system should answer.

    Returns:
    answer (str): The generated answer to the question.
    """
    docs = load_docs(url)            # Load documents from the URL
    chunks = get_text_chunks(docs)   # Split documents into text chunks
    vectorstore, retriever = get_vector_store(chunks)  # Create vector store and retriever
    ragchain = get_rag_chain(retriever)  # Create RAG chain
    answer = ragchain.invoke(question)   # Get the answer from the RAG chain
    del vectorstore  # Clean up the vector store to free resources

    return answer
