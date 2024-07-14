import requests
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA  
import os

from dotenv import load_dotenv
load_dotenv()

x = requests.get('http://127.0.0.1:8000/api/info')
with open('documents/1.txt', 'w') as f:
    f.write(x.text)
    
text_loader= TextLoader('documents/1.txt')
data = text_loader.load()
print('Number of documents: ',len(data))


text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
data = text_splitter.split_documents(data)
print('Number of chunks: ',len(data))

embeddings=OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
print('Embeddings loaded : ', embeddings, "\n\n")

try:
    vectorstore = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
except Exception as e:
    print(e)
    print("Error in creating retriever")

llm = OpenAI()
print('LLM created : ', llm)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


while True:
    print("\n Chat with me! Type 'exit' to quit.\n")
    query = input("tanyakan sesuai konteks bang: ")
    if query == "exit":
        break
    response_qa = qa.invoke(query)
    
    print("QA response: ", response_qa)

