import os
import streamlit as st
from langsmith import traceable
from writerai import Writer
from typing import List
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeVectorStore

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Writer AI RAG Demo",
    layout="centered"
)

# Initialize session state for current question if it doesn't exist
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Initialize constants and clients
MODEL_NAME = "palmyra-x-004"
MODEL_PROVIDER = "writer"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an AI assistant specializing in Writer’s products and documentation. Use only the provided retrieved context to answer questions related to Writer’s tools, features, or policies. 
If the context does not provide enough information, respond with "I don’t know." Keep your answers concise and maintain a professional tone.
"""
QUERY_REWRITE_PROMPT = """
You are a helpful assistant that rewrites user questions into standalone, well-formed search queries for a retrieval-augmented AI assistant.
This assistant helps users learn about Writer AI's products, services, capabilities, customers, and use cases.

When rewriting:
- Always expand ambiguous references like "writer" into "Writer AI"
- Preserve the user's intent while making it more specific
- Focus on the core question being asked
- Avoid greetings or casual phrasing — treat it like a search query
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
index_name = "writer-docs-gpt"

# Get vector database retriever
@st.cache_resource
def get_vector_db_retriever():
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embd)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

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

@traceable(run_type="llm", name="rewrite_query")
def rewrite_query(original_question: str) -> str:
    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": f"Rewrite this query for better search: {original_question}"}
    ]
    return call_writer(messages).choices[0].message.content.strip()
    
@traceable(run_type="chain")
def langsmith_rag(question: str):
    rewritten_query = rewrite_query(question)
    with st.spinner("Retrieving relevant documents..."):
        documents = retrieve_documents(rewritten_query)
    
    with st.spinner("Generating response with Writer's Palmyra..."):
        response = generate_response(rewritten_query, documents)
    
    # Extract source URLs from document metadata
    sources = []
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    
    return response.choices[0].message.content
# ----------------------------- Streamlit-Interface-----------------------------
with st.container():
    col1, col2, col3 = st.columns([1, 12, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Writer AI RAG Application</h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center;'>Ask questions about Writer's products, services, and capabilities</h6>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>To get started, try one of these example questions:</p>", unsafe_allow_html=True)
        
with st.container():
    col1, col2, col3 = st.columns([1,9,1])
    with col2:
        # Show the image by referencing the PNG file.
        # Make sure 'writer-rag-flow.png' is in the same directory or in an accessible path.
        st.image("writer-rag-flow-updated.png", use_container_width=True, caption="Writer AI RAG Flow Architecture")
        
# Example questions as buttons
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("How does Writer's Palmyra compare to other generative AI models in the enterprise space?"):
            st.session_state.current_question = "How does Writer's Palmyra LLM compare to other generative AI models in the enterprise space?"
        if st.button("What are the successful enterprise use cases for Writer's AI platform?"):
            st.session_state.current_question = "What are the successful enterprise use cases for Writer's AI platform?"

    with col2:
        if st.button("How does Writer handle hallucinations in enterprise contexts?"):
            st.session_state.current_question = "How does Writer handle hallucinations in enterprise contexts compared to other commercial LLMs?"
        if st.button("How is Writer incorporating agentic AI into Palmyra's enterprise workflow roadmap?"):
            st.session_state.current_question = "How is Writer incorporating agentic AI into Palmyra's enterprise workflow roadmap?"

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

    
