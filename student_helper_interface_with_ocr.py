#student_helper_interface_with_ocr.py(but not working)

import streamlit as st
import requests
import os
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# OCR libraries
import pytesseract
from pdf2image import convert_from_path

load_dotenv()

# ------------------ CONFIG TESSERACT ------------------
# Make sure Tesseract is installed at this path (or change it to your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ------------------ CUSTOM EMBEDDINGS CLASS ------------------
class LocalOllamaEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name
        self.url = "http://127.0.0.1:11434/api/embed"

    def embed_documents(self, texts):
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text):
        payload = {"model": self.model_name, "input": text}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["embeddings"][0]
# -------------------------------------------------------------

# ------------------ OCR PDF EXTRACTION ------------------
def extract_pdf_content(pdf_path):
    """Extracts text from a PDF. Uses OCR if the PDF is scanned."""
    try:
        # 1. Try normal text extraction
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Check if there is any actual text
        text_content = "".join([doc.page_content for doc in docs])
        
        # If there is meaningful text, just return it
        if len(text_content.strip()) > 50: 
            return docs
        
        # 2. If no text found, it's a scanned PDF. Run OCR.
        print("No text found, running OCR...")
        ocr_docs = []
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Run Tesseract on each image
                text = pytesseract.image_to_string(image)
                
                # Create a Document object for LangChain
                doc = Document(page_content=text, metadata={"page": i+1, "source": pdf_path})
                ocr_docs.append(doc)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            
        return ocr_docs
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

# -------------------------------------------------------------

st.set_page_config(page_title="Student Helper", page_icon="📚")
st.title("📚 Student Document Helper")

# ---------- CRITICAL SECURITY FIX: User Login ----------
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

with st.sidebar:
    st.header("🔐 Private Space")
    user_id = st.text_input("Enter your unique ID (e.g., your email)", value=st.session_state.user_id)
    
    if st.button("Set Private ID"):
        st.session_state.user_id = user_id
        st.success(f"Private space set for: {user_id}")

# If no user_id is set, block the app
if not st.session_state.user_id:
    st.warning("⚠️ Please enter a unique ID in the sidebar to start. Your data will be completely isolated.")
    st.stop()  # Stops the app from running further

# ---------- SIDEBAR: UPLOAD & SETTINGS (Fixed Chunk Size) ----------
with st.sidebar:
    st.header(f"Upload for {st.session_state.user_id}")
    
    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Upload to Pinecone"):
        if uploaded_files:
            with st.spinner("Processing... (This may take time for scanned PDFs)"):
                try:
                    all_texts = []
                    for uploaded_file in uploaded_files:
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Use the custom OCR-aware loader
                        docs = extract_pdf_content(uploaded_file.name)
                        
                        # FIXED CHUNK SIZE TO 1000 (No slider)
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, 
                            chunk_overlap=200
                        )
                        texts = text_splitter.split_documents(docs)
                        all_texts.extend(texts)
                        os.remove(uploaded_file.name)
                    
                    embeddings = LocalOllamaEmbeddings(model_name='nomic-embed-text-v2-moe:latest')
                    
                    # CRITICAL: Use the user_id as the namespace!
                    vector_store = PineconeVectorStore(
                        embedding=embeddings,
                        index_name='student-note-store',
                        namespace=st.session_state.user_id  # <--- SECURITY ISOLATION
                    )
                    
                    ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in all_texts]
                    vector_store.add_documents(all_texts, ids=ids)
                    st.success(f"✅ Uploaded to private space: {st.session_state.user_id}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please upload a PDF.")

# ---------- CHAT INTERFACE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store as list of dicts

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if user_input := st.chat_input("Ask a question about your notes..."):
    # Add user message to session state
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                embeddings = LocalOllamaEmbeddings(model_name='nomic-embed-text-v2-moe:latest')
                
                # CRITICAL: Search ONLY within the user's namespace!
                vector_store = PineconeVectorStore(
                    embedding=embeddings,
                    index_name='student-note-store',
                    namespace=st.session_state.user_id  # <--- SECURITY ISOLATION
                )
                
                retriever = vector_store.as_retriever(search_kwargs={'k': 4})

                prompt = ChatPromptTemplate.from_template(
                    """You are a helpful study assistant. Answer the question based on the provided context.
                    If you don't have the relevant answer - just say 'i dont have information about this'. *don't hallucinate*
                    context:{context}
                    query/question:{question}
                    answer:
                    """
                )

                llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')  # Valid model
                parser = StrOutputParser()
                rag_chain = (
                    {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm | parser
                )

                response = rag_chain.invoke(user_input)
                
                # Display AI message
                st.markdown(response)
                
                # Add AI message to session state (MUST be a dict!)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Oops! An error occurred: {e}")