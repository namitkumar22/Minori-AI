import dotenv
import os
import time
import pickle
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()

class GenerateSolution:
    def __init__(self):
        os.environ["GROQ_API"] = os.getenv("GROQ_API")
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API"),
            model="openai/gpt-oss-120b",
            temperature=0
        )
        self.embeddings = None
        self.vectors = None

    def _get_embeddings(self):
        """Always use HuggingFace embeddings."""
        print("🔗 Using HuggingFace embeddings...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def generate(self, crop, disease):  # Only for testing without context
        question = f"Please tell the solution according to the official methods suggested by the government of India for the following crop and disease. Crop: {crop}, Disease: {disease}"
        response = self.llm.invoke(question)
        print(response.content)

    def process_documents(self):
        try:
            if self.embeddings is None:
                self.embeddings = self._get_embeddings()

            if not os.path.exists("Vectors"):
                loader = PyPDFDirectoryLoader("Data")
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs[:])
                self.vectors = FAISS.from_documents(final_documents, self.embeddings)
                self.vectors.save_local("Vectors")
                print("✅ FAISS index built with HuggingFace embeddings.")
            else:
                self.vectors = FAISS.load_local("Vectors", self.embeddings, allow_dangerous_deserialization=True)
                print("📂 Loaded existing FAISS index.")

            return True
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return False

    def generate_with_context(self, crop, disease):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert agriculture consultant. Use the provided context to answer the question accurately. If the context does not contain relevant information, respond with 'I am not aware about this'."),
            ("user", "Based on the following context, please provide information about the crop and disease. If the context does not contain relevant information, respond with 'I am not aware about this'.\n\nContext: {context}\n\nQuestion: {input}. Also avoid using tabular structure just dirrect bullet point structure.")
        ])

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retriever = self.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        questions = f"Please tell the solution according to the official methods present in the context for the following crop and disease. Crop: {crop}, Disease: {disease}"

        try:
            start_time = time.process_time()
            self.response = retrieval_chain.invoke({'input': questions})
            self.processing_time = time.process_time() - start_time

            print("Answer:", self.response["answer"])
            print("Processing time:", self.processing_time, "seconds")

        except Exception as e:
            print("❌ Error during generate_with_context:", str(e))


GS = GenerateSolution()
if GS.process_documents():
    GS.generate_with_context("wheat", "yellow rust")
