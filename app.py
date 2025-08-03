import os
import re
import io
import json
import time
import random
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, InternalServerError

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'

# --- Configuration ---
# Read API keys and connection strings from environment variables set by Azure App Service.
# These values are pulled from your App Service's 'Configuration' -> 'Application settings'.
google_api_key = os.getenv("GOOGLE_API_KEY")
azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# --- Global Azure Blob Service Client ---
blob_service_client = None
try:
    if azure_storage_connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
        print("Azure Blob Service Client initialized.")
    else:
        print("WARNING: AZURE_STORAGE_CONNECTION_STRING not set. Blob storage features will not work.")
except Exception as e:
    print(f"ERROR: Could not initialize Azure Blob Service Client: {e}")

# Global variables for RAG pipeline components
rag_pipeline_initialized = False
current_video_id = None
vector_store = None
retriever = None
# rag_chain will be initialized dynamically in initialize_rag_pipeline

# --- Initialize Embeddings and LLMs Globally ---
# It's better to initialize both LLMs explicitly if they are for different purposes.
llm_rag = None
llm_direct_chat = None
embeddings = None

try:
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please configure it in Azure App Service settings.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)

    # LLM for YouTube Q&A (RAG mode)
    llm_rag = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, api_key=google_api_key)
    print("LLM for RAG (gemini-1.5-flash) initialized.")

    # LLM for Direct Chat (general purpose)
    llm_direct_chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2, api_key=google_api_key)
    print("LLM for Direct Chat (gemini-2.0-flash-lite) initialized.")

    print("Embeddings and LLM models initialized globally.")
except Exception as e:
    print(f"FATAL ERROR: Could not initialize Google Generative AI models. "
          f"Please ensure GOOGLE_API_KEY is correctly set and valid in your Azure App Service configuration: {e}")

# Define prompt template globally (it can be reused for RAG)
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    Context: {context}
    Question: {question}
    """,
    input_variables = ['context', 'question']
)

# --- Helper Functions for Azure Blob Storage ---

def get_blob_container_client(container_name):
    if not blob_service_client:
        return None
    try:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            print(f"Container '{container_name}' created.")
        return container_client
    except Exception as e:
        print(f"Error getting/creating container '{container_name}': {e}")
        return None

def save_transcript_to_blob(video_id, transcript_text):
    container_client = get_blob_container_client("transcripts")
    if not container_client: return False
    try:
        blob_client = container_client.get_blob_client(f"{video_id}.txt")
        blob_client.upload_blob(transcript_text, overwrite=True)
        print(f"Transcript for {video_id} saved to Blob Storage.")
        return True
    except Exception as e:
        print(f"Error saving transcript for {video_id} to Blob Storage: {e}")
        return False

def load_transcript_from_blob(video_id):
    container_client = get_blob_container_client("transcripts")
    if not container_client: return None
    try:
        blob_client = container_client.get_blob_client(f"{video_id}.txt")
        if blob_client.exists():
            download_stream = blob_client.download_blob()
            print(f"Transcript for {video_id} loaded from Blob Storage.")
            return download_stream.readall().decode('utf-8')
        return None
    except Exception as e:
        print(f"Error loading transcript for {video_id} from Blob Storage: {e}")
        return None

def save_faiss_index_to_blob(faiss_index_obj, video_id):
    container_client = get_blob_container_client("faiss-indexes")
    if not container_client:
        return False

    temp_dir = f"temp_faiss_index_{video_id}" # Use video_id for unique temp dir
    os.makedirs(temp_dir, exist_ok=True)
    index_prefix = os.path.join(temp_dir, "index")

    try:
        # Save FAISS index locally (creates index.faiss and index.pkl)
        faiss_index_obj.save_local(temp_dir, index_name="index") # Save directly to temp_dir with a name

        # Upload both .faiss and .pkl files
        faiss_file_path = os.path.join(temp_dir, "index.faiss")
        pkl_file_path = os.path.join(temp_dir, "index.pkl")

        if os.path.exists(faiss_file_path):
            blob_client_faiss = container_client.get_blob_client(f"faiss_index_{video_id}.faiss")
            with open(faiss_file_path, "rb") as f:
                blob_client_faiss.upload_blob(f.read(), overwrite=True)
            print(f"FAISS index data for {video_id} saved to Blob Storage.")
        else:
            print(f"Warning: .faiss file not found at {faiss_file_path}")
            return False

        if os.path.exists(pkl_file_path):
            blob_client_pkl = container_client.get_blob_client(f"faiss_index_{video_id}.pkl")
            with open(pkl_file_path, "rb") as f:
                blob_client_pkl.upload_blob(f.read(), overwrite=True)
            print(f"FAISS index metadata for {video_id} saved to Blob Storage.")
        else:
            print(f"Warning: .pkl file not found at {pkl_file_path}")
            return False

        return True
    except Exception as e:
        print(f"Error saving FAISS index for {video_id} to Blob Storage: {e}")
        return False
    finally:
        # Cleanup: remove temporary files and directory
        if os.path.exists(faiss_file_path):
            os.remove(faiss_file_path)
        if os.path.exists(pkl_file_path):
            os.remove(pkl_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        

def load_faiss_index_from_blob(video_id, embeddings_model):
    container_client = get_blob_container_client("faiss-indexes")
    if not container_client:
        return None

    temp_dir = f"temp_faiss_index_load_{video_id}" # Use video_id for unique temp dir
    os.makedirs(temp_dir, exist_ok=True)
    temp_faiss_path = os.path.join(temp_dir, "index.faiss")
    temp_pkl_path = os.path.join(temp_dir, "index.pkl")

    try:
        # Download .faiss file
        blob_client_faiss = container_client.get_blob_client(f"faiss_index_{video_id}.faiss")
        if blob_client_faiss.exists():
            with open(temp_faiss_path, "wb") as f:
                download_stream = blob_client_faiss.download_blob()
                f.write(download_stream.readall())
        else:
            print(f"FAISS .faiss file for {video_id} not found in Blob Storage.")
            return None

        # Download .pkl file
        blob_client_pkl = container_client.get_blob_client(f"faiss_index_{video_id}.pkl")
        if blob_client_pkl.exists():
            with open(temp_pkl_path, "wb") as f:
                download_stream = blob_client_pkl.download_blob()
                f.write(download_stream.readall())
        else:
            print(f"FAISS .pkl file for {video_id} not found in Blob Storage.")
            # It's possible to load without .pkl if only index is needed, but for full FAISS.load_local, it's usually required.
            # For Langchain's FAISS, the .pkl is crucial for docstore and embeddings.
            return None

        # Load FAISS index from the temporary directory
        faiss_vector_store = FAISS.load_local(temp_dir, embeddings_model, allow_dangerous_deserialization=True)
        print(f"FAISS index for {video_id} loaded from Blob Storage.")
        return faiss_vector_store
    except Exception as e:
        print(f"Error loading FAISS index for {video_id} from Blob Storage: {e}")
        return None
    finally:
        # Cleanup: remove temporary files and directory
        if os.path.exists(temp_faiss_path):
            os.remove(temp_faiss_path)
        if os.path.exists(temp_pkl_path):
            os.remove(temp_pkl_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

# --- Helper Function to Extract Video ID ---
def get_youtube_video_id(url):
    """
    Extracts the YouTube video ID from a given URL.
    Supports various YouTube URL formats.
    """
    if "youtube.com/watch?v=" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    elif "youtu.be/" in url:
        match = re.search(r"youtu\.be/([a-zA-Z0-9_-]{11})", url)
    else:
        match = None

    if match:
        return match.group(1)
    return None

# --- Function to Initialize RAG Pipeline (for video Q&A) ---
def initialize_rag_pipeline(video_id):
    global rag_pipeline_initialized, current_video_id, vector_store, retriever, rag_chain, llm_rag

    print(f"Initializing RAG pipeline for video ID: {video_id}")
    transcript_text = ""
    faiss_loaded = False

    if not embeddings:
        return False, "Embedding model not initialized. Please check API key."

    # Try loading FAISS index from Blob Storage first
    loaded_vector_store = load_faiss_index_from_blob(video_id, embeddings)
    if loaded_vector_store:
        vector_store = loaded_vector_store
        faiss_loaded = True
        print("FAISS index loaded from Blob Storage, skipping transcript fetch and embedding generation.")
    else:
        # If FAISS index not found, try loading transcript from Blob Storage
        transcript_text = load_transcript_from_blob(video_id)
        if transcript_text:
            print("Transcript loaded from Blob Storage.")
        else:
            # If transcript not found in Blob, fetch from YouTube
            print("Transcript not found in Blob Storage, fetching from YouTube...")
            try:
                ytt_api = YouTubeTranscriptApi()
                fetched_transcript = ytt_api.fetch(video_id, languages=["en", "es", "fr", "de"])
                transcript_list_raw = fetched_transcript.to_raw_data()
                transcript_text = " ".join(chunk["text"] for chunk in transcript_list_raw)
                print("Transcript fetched successfully from YouTube!")
                save_transcript_to_blob(video_id, transcript_text) # Save to Blob after fetching

            except NoTranscriptFound:
                print(f"Error: No transcripts found for video ID: {video_id}.")
                return False, "No transcripts found for this video. It might not have captions."
            except TranscriptsDisabled:
                print(f"Error: Transcripts are disabled for video ID: {video_id}.")
                return False, "Transcripts are disabled for this video."
            except Exception as e:
                print(f"An unexpected error occurred while fetching transcript: {e}. "
                      f"If running on a cloud provider (like Azure), YouTube might be blocking your IP. "
                      f"Consider using a proxy or pre-uploading transcripts.")
                return False, f"An error occurred while fetching transcript: {e}. " \
                               "If running on a cloud provider (like Azure), YouTube might be blocking your IP. " \
                               "Consider using a proxy (not currently configured) or pre-uploading transcripts to Blob Storage."

        if not transcript_text:
            return False, "Could not retrieve transcript."

        print("Splitting transcript into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])
        print(f"Transcript split into {len(chunks)} chunks.")

        print("Generating embeddings and building vector store (FAISS)...")
        try:
            vector_store = FAISS.from_documents(chunks, embeddings)
            print("Vector store created successfully!")
            save_faiss_index_to_blob(vector_store, video_id) # Save FAISS index to Blob
        except Exception as e:
            print(f"Error creating vector store: {e}. This might be due to API key issues, "
                  f"network problems, or environment configuration.")
            return False, f"Error creating vector store: {e}"

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Define rag_chain using the specific 'llm_rag' for RAG operations
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm_rag # Use llm_rag here
        | StrOutputParser()
    )

    rag_pipeline_initialized = True
    current_video_id = video_id # Set video_id for video Q&A mode
    return True, "Pipeline initialized successfully."

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the homepage for YouTube URL input."""
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Receives YouTube URL, processes it, and initializes the RAG pipeline.
    Redirects to the Q&A page on success.
    """
    video_url = request.form.get('video_url')
    if not video_url:
        return render_template('index.html', error="Please enter a YouTube video URL.")

    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return render_template('index.html', error="Invalid YouTube URL. Please check the format.")

    # Initialize the RAG pipeline for the given video
    success, message = initialize_rag_pipeline(video_id)

    if success:
        session['chat_mode'] = 'video_qa' # Set chat mode
        session['current_video_id'] = video_id # Store video_id for video Q&A mode
        return redirect(url_for('qa_page'))
    else:
        return render_template('index.html', error=message)

@app.route('/qa')
def qa_page():
    """Renders the Q&A page for video transcripts."""
    # Ensure llm_rag is available for this mode
    if not llm_rag:
        return render_template('index.html', error="RAG model not initialized. Please check server logs.")

    if session.get('chat_mode') != 'video_qa' or not rag_pipeline_initialized:
        return redirect(url_for('index')) # Redirect if not in video Q&A mode or pipeline not initialized
    video_id = session.get('current_video_id')
    return render_template('qa.html', video_id=video_id)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Receives a question via AJAX for video Q&A, uses the RAG pipeline to get an answer,
    and returns it as JSON. Includes retry logic for rate limits.
    """
    if session.get('chat_mode') != 'video_qa' or not rag_pipeline_initialized:
        return jsonify({"error": "No video processed or incorrect chat mode. Please go back to the homepage."}), 400

    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    print(f"Received video Q&A question: {question}")
    max_retries = 3
    base_delay = 1 # seconds
    for attempt in range(max_retries):
        try:
            # rag_chain is already initialized with llm_rag in initialize_rag_pipeline
            answer = rag_chain.invoke(question)
            print(f"Generated answer: {answer[:100]}...")
            return jsonify({"answer": answer})
        except (ResourceExhausted, DeadlineExceeded, InternalServerError) as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"API Error (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            if attempt == max_retries - 1:
                print("Max retries reached for API error. Returning fallback error.")
                return jsonify({
                    "error": "The AI model is currently unavailable or experiencing high load. "
                             "Please try again in a few moments."
                }), 503
        except Exception as e:
            print(f"Unexpected error generating answer: {e}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    return jsonify({"error": "Failed to get an answer after multiple retries."}), 500

# --- Direct AI Chat Routes ---
@app.route('/direct_qa')
def direct_qa_page():
    """Renders the direct AI Q&A page."""
    # Ensure llm_direct_chat is available for this mode
    if not llm_direct_chat:
        return render_template('index.html', error="Direct chat model not initialized. Please check server logs.")

    session['chat_mode'] = 'direct_qa' # Set chat mode
    session['current_video_id'] = 'direct_chat' # Placeholder for direct chat saving
    return render_template('direct_qa.html')

@app.route('/ask_direct_question', methods=['POST'])
def ask_direct_question():
    """
    Receives a question via AJAX for direct AI chat, uses the LLM to get an answer,
    and returns it as JSON. Includes retry logic for rate limits.
    """
    if session.get('chat_mode') != 'direct_qa':
        return jsonify({"error": "Incorrect chat mode. Please go back to the homepage."}), 400

    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if not llm_direct_chat: # Ensure the model is available
        return jsonify({"error": "Direct chat AI model is not initialized."}), 500

    print(f"Received direct AI question: {question}")
    max_retries = 3
    base_delay = 1 # seconds
    for attempt in range(max_retries):
        try:
            ai_response = llm_direct_chat.invoke(question) # Use llm_direct_chat here
            answer = ai_response.content
            print(f"Generated direct AI answer: {answer[:100]}...")
            return jsonify({"answer": answer})
        except (ResourceExhausted, DeadlineExceeded, InternalServerError) as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"API Error (Attempt {attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            if attempt == max_retries - 1:
                print("Max retries reached for API error. Returning fallback error.")
                return jsonify({
                    "error": "The AI model is currently unavailable or experiencing high load. "
                             "Please try again in a few moments."
                }), 503
        except Exception as e:
            print(f"Unexpected error generating direct AI answer: {e}")
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    return jsonify({"error": "Failed to get an answer after multiple retries."}), 500

@app.route('/save_chat_history', methods=['POST'])
def save_chat_history():
    """
    Receives chat history from the frontend and saves it to Blob Storage.
    The filename will reflect the chat mode (video_qa or direct_qa).
    """
    chat_data = request.json.get('chat_history')
    chat_mode = session.get('chat_mode', 'unknown_mode')
    identifier = session.get('current_video_id', 'unknown_id')

    if not chat_data:
        return jsonify({"error": "No chat history provided."}), 400

    container_client = get_blob_container_client("chat-histories")
    if not container_client:
        return jsonify({"error": "Blob storage not configured or accessible for chat histories."}), 500

    try:
        filename = f"{chat_mode}_chat_history_{identifier}_{os.urandom(4).hex()}.json"
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(json.dumps(chat_data, indent=2), overwrite=True, encoding='utf-8')
        print(f"Chat history for {identifier} ({chat_mode}) saved as {filename} to Blob Storage.")
        return jsonify({"message": "Chat history saved successfully!"}), 200
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return jsonify({"error": f"Failed to save chat history: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)