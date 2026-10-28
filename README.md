📚 Your AI PDF Assistant — Multi-PDF RAG Chatbot
An AI-powered PDF question-answering application built with Streamlit, LangChain, Groq, Hugging Face embeddings, and FAISS. Upload one or more PDFs, create a shared searchable knowledge base, and ask questions grounded in the documents.

✨ Features
Multi-PDF upload: Select multiple PDF documents in the Streamlit sidebar.
Unified document index: Split and embed all uploaded PDFs together in a single FAISS vector store.
Semantic retrieval: Find relevant passages across documents for each question.
Grounded responses: Generate answers with Groq's openai/gpt-oss-120b model using retrieved PDF excerpts.
Source transparency: Expand retrieved sources to view filenames and page numbers.
Session chat history: View previous questions and responses during the session.
Index management: Avoid reprocessing identical files and clear the knowledge base when needed.
🛠️ Tech Stack
Component	Technology
User interface	Streamlit
RAG orchestration	LangChain
LLM	Groq API (openai/gpt-oss-120b)
PDF extraction	PyPDFLoader / PyPDF
Text splitting	RecursiveCharacterTextSplitter
Embeddings	sentence-transformers/all-MiniLM-L12-v2
Vector search	FAISS
Configuration	python-dotenv
🧠 How It Works
Upload multiple PDFs (Streamlit)
              ↓
       Extract PDF text
              ↓
  Split into overlapping chunks
     (1,000 chars, 100 overlap)
              ↓
    Hugging Face embeddings
              ↓
     Shared FAISS vector index
              ↓
        User question
              ↓
 Retrieve top 4 relevant chunks
              ↓
 Groq LLM + retrieved context
              ↓
 Answer + source PDF/page references
🚀 Run Locally
Prerequisites
Python and Pipenv installed
A Groq API key
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
Replace the placeholders with your GitHub username and repository name.

2. Install dependencies
If your project contains a Pipfile:

pipenv install
Otherwise, install the packages:

pipenv install streamlit python-dotenv langchain-community langchain-core langchain-groq langchain-huggingface langchain-text-splitters pypdf sentence-transformers faiss-cpu
3. Configure the API key
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
Never commit .env or API keys to GitHub. Add .env to your .gitignore.

4. Launch the application
pipenv run streamlit run app.py
Open the local URL shown in the terminal (typically http://localhost:8501).

💬 Usage
Upload one or more PDFs using the sidebar uploader.
Click Process documents to create the combined index.
Ask questions in the chat input, such as:
“Summarize the key findings in these PDFs.”
“What are the differences between the two documents?”
“Explain the main idea of the research paper.”
Open Retrieved sources to inspect the source filenames and pages.
Click Clear knowledge base to remove the current index and chat history.
📁 Suggested Project Structure
AI-RAG-Chatbot/
├── app.py
├── README.md
├── assets/
│   └── chatbot-demo.png
├── Pipfile                 # if using Pipenv
├── Pipfile.lock            # if generated
├── .gitignore
└── .env                    # local only; do not commit
⚠️ Current Limitations
PDFs must contain extractable text; scanned/image-only PDFs require OCR.
The FAISS index is stored in the current Streamlit session rather than persisted to disk.
Chat messages are displayed, but prior messages are not included in the model's prompt; follow-up questions may lack conversational context.
Answers depend on retrieved excerpts, so broad document summaries may not cover every section.
🔮 Future Improvements
Conversational memory for follow-up questions
Persistent FAISS storage and index reuse
OCR support for scanned PDFs
Optional document filtering and more robust citations
Automated RAG evaluation and tests
Built as a hands-on project exploring Retrieval-Augmented Generation (RAG) and document-grounded AI applications.
