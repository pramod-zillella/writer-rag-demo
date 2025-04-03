import os
import streamlit as st
from langsmith import traceable
from writerai import Writer
from typing import List
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone

# Set page configuration
st.set_page_config(
    page_title="Writer AI RAG Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for current question if it doesn't exist
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Initialize constants and clients
MODEL_NAME = "palmyra-x-004"
MODEL_PROVIDER = "writer"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an AI assistant specializing in Writer’s products and documentation. Use only the provided retrieved context to answer questions related to Writer’s tools, features, or policies. 
If the context does not provide enough information, respond with "I don’t know." Keep your answers concise (5 sentences maximum) and maintain a professional tone.
"""

# Initialize clients and services
@st.cache_resource
def initialize_clients():
    writer_client = Writer()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    embd = OpenAIEmbeddings(model="text-embedding-3-large")
    return writer_client, pc, embd

writer_client, pc, embd = initialize_clients()

# Define the Pinecone index name
index_name = "writer-docs"

# Get vector database retriever
@st.cache_resource
def get_vector_db_retriever():
    vectorstore = LangChainPinecone.from_existing_index(index_name, embd)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Create retriever instance
retriever = get_vector_db_retriever()

@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_writer(messages)

@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_writer(messages: List[dict]) -> str:
    return writer_client.chat.chat(
        model=MODEL_NAME,
        messages=messages,
    )

@traceable(run_type="chain")
def langsmith_rag(question: str):
    with st.spinner("Retrieving relevant documents..."):
        documents = retrieve_documents(question)
    
    with st.spinner("Generating response with Writer's Palmyra..."):
        response = generate_response(question, documents)
    
    # Extract source URLs from document metadata
    sources = []
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    
    return response.choices[0].message.content

# # Create sidebar with information
# with st.sidebar:
#     st.header("About this Demo")
#     st.markdown("""
#     This application demonstrates a RAG implementation using:
    
#     - **Writer's Palmyra LLM** for generation
#     - **Pinecone** as the vector database
#     - **LangChain** for the RAG pipeline
#     - **LangSmith** for tracing and evaluation
    
#     The knowledge base consists of Writer's documentation, allowing the model to answer questions about Writer's products and services.
#     """)
    
#     st.divider()
#     st.markdown(f"**Model**: {MODEL_NAME}")
#     st.markdown(f"**App Version**: {APP_VERSION}")



with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Writer AI RAG Application</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Ask questions about Writer's products, services, and capabilities</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>To get started, try one of these example questions:</p>", unsafe_allow_html=True)

# Example questions as buttons
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("How does Writer's Palmyra compare to other generative AI models in the enterprise space?"):
            st.session_state.current_question = "How does Writer's Palmyra LLM compare to other generative AI models in the enterprise space?"
        if st.button("Explain Writer's approach to context window optimization?"):
            st.session_state.current_question = "Explain Writer's approach to context window optimization and how it compares to competitors?"
        if st.button("How does Writer's context-aware text splitting enhance the processing of lengthy documents?"):
            st.session_state.current_question = "How does Writer's context-aware text splitting enhance the processing of lengthy documents in AI applications?"

    with col2:
        if st.button("How does Writer handle hallucinations in enterprise contexts?"):
            st.session_state.current_question = "How does Writer handle hallucinations in enterprise contexts compared to other commercial LLMs?"
        if st.button("How is Writer incorporating agentic AI into Palmyra's enterprise workflow roadmap?"):
            st.session_state.current_question = "How is Writer incorporating agentic AI into Palmyra's enterprise workflow roadmap?"
        if st.button("What are the successful enterprise use cases for Writer's AI platform?"):
            st.session_state.current_question = "What are the successful enterprise use cases for Writer's AI platform?"

# st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface logic
if st.session_state.current_question:
    prompt = st.session_state.current_question
    st.session_state.current_question = ""  # Clear it after use
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = langsmith_rag(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Text input for custom questions
prompt = st.chat_input("Ask a question about Writer")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = langsmith_rag(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})