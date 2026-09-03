# RAG_based_chatbot.py

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
#### API keyss

###################################
load_dotenv()
llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')  # ✅ CHANGED: gemini-3.6-flash → gemini-1.5-flash


parser = StrOutputParser()
loader = DirectoryLoader(
    path='data1',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()
print('document loaded from the folder...')
# for documents in docs:
#     print(documents.metadata)


# now spliting and chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
texts = text_splitter.split_documents(docs)
print('splitted the texts....')

## now embeddings
print('adding the embeddings..')
# query = 'what is deep larning ?'
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text-v2-moe:latest')
# query_vector = embeddings.embed_query(query)
print('embeddings added...')
# print("length of text  vector is :",len(query_vector))




### establishing picecone vector store #####################
import hashlib
print('establishing vector store...')
from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name='student-note-store'
)
print('1111111')
ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in texts]
print('222222')
# vector_store.add_documents(texts, ids =ids)
print("✅ Documents added to Pinecone")
###########################################################


### now adding retrivers...
retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        'k':4,
        'fetch_k':20,
        'lambda_mult':0.5
    }
)
## Adding memory
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# memory = InMemoryChatMessageHistory()


# creating the themplate and prompt temolate
from langchain_core.prompts import ChatPromptTemplate

template="""You are a helpful study assistant. Directly start answering on the provided context from the student's notes.You are a helpful study assistant. Answer the question based on the provided context from the student's notes.
if you dont have the relevent answer - just 'say i dont have information about this'. *dont halucinate*

context:{context}

query/question:{question}

answer:
"""
prompt = ChatPromptTemplate.from_template(template)  # ✅ CHANGED: removed template= parameter




##creating final RAG chain
from langchain_core.runnables import RunnablePassthrough
query = 'what is stonechip ?'



rag_chain = (
    {'context':retriever, 'question':RunnablePassthrough()} | prompt |llm | parser
)
print('chain is created....')

# chain_with_history = RunnableWithMessageHistory(
#     rag_chain,
#     # Use the memory object we created
#     lambda session_id: memory,
#     input_messages_key="question",
#     history_messages_key="history" 
# )

## testing
print('\n📚 RAG Chain ready for queries!')
answer = rag_chain.invoke(query)

print('\n\n')
print(f"Answer : {answer}")


