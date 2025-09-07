import dotenv
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import pickle

dotenv.load_dotenv()
class GenerateSolution:
    def __init__(self):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API"),
            model="openai/gpt-oss-120b",
            temperature=0
        )

    def generate(self, crop, disease): # Only for testing without context 

        question = f"Please tell the solution according to the official methods suggested by the government of India for the following crop and disease. Crop: {crop}, Disease: {disease}"
        response = self.llm.invoke(question)
        print(response.content)

    def process_documents(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            if not os.path.exists("Backend\\Vectors"):
                loader = PyPDFDirectoryLoader("Backend\Data")
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs[:])
                self.vectors = FAISS.from_documents(final_documents, self.embeddings)
                self.vectors.save_local("Backend\\Vectors")
                
            return True
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return False
        
    def generate_with_context(self, crop, disease):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agriculture consultant. Use the provided context to answer the question accurately. If the context does not contain relevant information, respond with 'I don't know'."),
            ("user", "Based on the following context, please provide information about the crop and disease. If the context does not contain relevant information, respond with 'I don't know'.\n\nContext: {context}\n\nQuestion: {input}")
        ])
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        vector_store = FAISS.load_local("Backend\\Vectors", self.embeddings, allow_dangerous_deserialization=True)

        retriever = vector_store.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        questions = "Please tell the solution according to the official methods present in the context for the following crop and disease. Crop: {}, Disease: {}".format(crop, disease)
        start_time = time.process_time()
        self.response = retrieval_chain.invoke({'input': questions})
        self.processing_time = time.process_time() - start_time

        print("Answer:", self.response["answer"])
        print("Processing time:", self.processing_time, "seconds")

GS = GenerateSolution()
if GS.process_documents():
    GS.generate_with_context("wheat", "yellow rust")