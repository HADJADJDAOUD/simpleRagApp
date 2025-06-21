import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

def main():
    # Set up paths - adjust these for your local environment
    pdf_path = "lekl101.pdf"  # Update this path to your PDF file
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable with the correct path to your PDF file")
        return
    
    print("Loading PDF document...")
    # Load the document
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    print(f"Loaded {len(data)} pages from PDF")
    
    # Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_documents(data)
    print(f"Created {len(chunks)} chunks")
    
    # Display first chunk for verification
    print("\nFirst chunk preview:")
    print(chunks[0].page_content[:200] + "..." if len(chunks[0].page_content) > 200 else chunks[0].page_content)
    
    # Initialize embedding model
    print("\nInitializing embedding model...")
    model_name = "thenlper/gte-large"
    embedding_model = FastEmbedEmbeddings(model_name=model_name)
    
    # Create vector store
    print("Creating vector store...")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
    
    # Initialize retriever
    retriever = db.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={'k': 4}
    )
    
    # Initialize Ollama with Gemma 3 4B
    print("Initializing Ollama with Gemma 3 4B...")
    llm = Ollama(
        model="gemma2:4b",  # Adjust model name if needed
        temperature=0.1,
    )
    
    # Create prompt template
    template = """
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

CONTEXT: {context}

QUESTION: {query}

ANSWER:
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    
    # Create the RAG chain
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    
    # Test queries
    queries = [
        "what is importance of I sell my dreams according to the author",
        "what is the author's name"
    ]
    
    print("\n" + "="*50)
    print("TESTING RAG SYSTEM")
    print("="*50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            response = chain.invoke(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        user_query = input("\nYour question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        try:
            response = chain.invoke(user_query)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Check if required packages are installed
    required_packages = [
        "langchain",
        "langchain-community", 
        "chromadb",
        "fastembed",
        "pypdf"
    ]
    
    print("RAG System with Ollama Gemma 3 4B")
    print("="*40)
    print("Required packages:", ", ".join(required_packages))
    print("Make sure you have Ollama installed and Gemma 3 4B model pulled")
    print("Run: ollama pull gemma2:4b")
    print("="*40)
    
    main()