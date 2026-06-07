import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

### LOADING ENVIRON VARIABLE


groq_api = st.secrets["GROQ_API_KEY"]

### PAGE SETTINGS
st.set_page_config(page_title="HR ASISSTANT", page_icon ="🤖" , layout="wide")
st.title("🤖 HR Policy Assistant")
st.write("Ask HR Policy Question")

### LOADING FILES
@st.cache_resource
def load_documents():
    data_dir = Path(__file__).resolve().parent / "data"

    documents = []
    pptx_files = list(data_dir.glob("*.pptx"))
    for pptx_file in pptx_files:
        loader = UnstructuredPowerPointLoader(str(pptx_file))
        documents.extend(loader.load())

    return documents


### CHUNKING
@st.cache_resource
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "", ""],
        chunk_size=500,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)
    return docs

### CREATING EMBEDDING + VECTOR STORE
@st.cache_resource
def create_vectorstore(docs):
    if not docs:
        raise ValueError("No document chunks available to build the vector store.")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    vector_store = FAISS.from_documents(docs, embedding_model)
    
    return vector_store

# ===============================
# STEP 7: LOAD LLM
# ===============================
@st.cache_resource
def load_llm():
    
    llm = ChatGroq(
        groq_api_key=groq_api,
        model="llama-3.3-70b-versatile"
    )
    
    return llm

# ===============================
# STEP 8: QUERY CLASSIFIER
# ===============================
def classify_query(query):
    
    prompt = f"""
    Classify the query into one category:
    
    1. policy_question
    2. general_question
    
    Query: {query}
    
    Return only category name.
    """
    
    result = llm.invoke(prompt).content
    return result.lower()

# ===============================
# STEP 9: MAIN AGENT FUNCTION
# ===============================
def hr_agent(query):
    intent = classify_query(query)
    
    if "policy" in intent:
        docs = vector_store.similarity_search(query, k=5)
        context = "\n".join([d.page_content for d in docs]) if docs else "No relevant documents found."
    else:
        context = "No relevant documents found."

    prompt = f"""You are an HR Assistant.
    Context: {context}
    Question: {query} """

    return llm.invoke(prompt).content
  

# ===============================
# STEP 10: BUILD PIPELINE
# ===============================
try:
    with st.spinner("Loading Documents..."):
        documents = load_documents()
        if not documents:
            raise ValueError("No .pptx files were found in the data directory.")

        docs = split_documents(documents)
        if not docs:
            raise ValueError("No document chunks were created from the loaded files.")

        vector_store = create_vectorstore(docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = load_llm()

    st.success("System Ready ✅")
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

# ===============================
# STEP 11: USER INPUT
# ===============================
query = st.text_input("Enter your question:")

if st.button("Ask"):

    if query:

        with st.spinner("Thinking..."):
            answer = hr_agent(query)

        st.subheader("Answer:")
        st.write(answer)

    else:
        st.warning("Please enter question.")

# ===============================
# STEP 12: SIDEBAR
# ===============================
st.sidebar.title("Instructions")
st.sidebar.write("""
 Asked Question Regarding Policies
                 
""")



