from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow all origins, methods, and headers
# This should be done ONCE after the app is initialized.
CORS(
    app,
    resources={r"/*": {"origins": "*"}}, # Allow all origins
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # Explicitly allow common methods, including OPTIONS
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"], # Allow common headers
    supports_credentials=True
)

# Import your PDF RAG functions
# Assuming pdf_rag_huggin.py exists in the same directory
from pdf_rag_huggin import (
    setup_huggingface_token,
    load_pdf_document,
    split_text_into_chunks,
    create_vector_store,
    setup_retriever,
    setup_llm,
    create_rag_chain,
    ask_question
)

# Temporary storage for pipeline state
STATE = {
    "db": None,
    "retriever": None,
    "llm": None,
    "chain": None
}

# Configure upload settings for the Flask app
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/setup_token', methods=['POST'])
def api_setup_token():
    token = request.json.get('token')
    print(f"Received token: {token}")
    if not token:
        return jsonify({'error': 'Token is required'}), 400

    try:
        # Pass the token directly to the function
        user = setup_huggingface_token(token=token, token_only=True)
        return jsonify({'message': 'Authenticated', 'user': user}), 200
    except Exception as e:
        print(f"Error setting token: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def api_upload_pdf():
    """Endpoint to upload and process a PDF document."""
    # Flask-CORS automatically handles OPTIONS for registered routes
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF is allowed.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"File saved to {filepath}")
    try:
        data = load_pdf_document(filepath)
        chunks = split_text_into_chunks(data)

        # Initialize and store the RAG pipeline components
        db = create_vector_store(chunks)
        retriever = setup_retriever(db)
        llm = setup_llm()
        chain = create_rag_chain(retriever, llm)
        STATE.update({'db': db, 'retriever': retriever, 'llm': llm, 'chain': chain})

        print(f"Pipeline initialized with {len(chunks)} chunks")
        return jsonify({'message': 'PDF processed', 'chunks': len(chunks)}), 200
    except Exception as e:
        print(f"Error processing PDF: {e}") # Log the error
        return jsonify({'error': f'Failed to process PDF: {str(e)}'}), 500
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted temporary file: {filepath}")

@app.route('/ask', methods=['POST'])
def api_ask():
    """Endpoint to ask a question about the uploaded PDF."""
    # Flask-CORS automatically handles OPTIONS for registered routes
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    chain = STATE.get('chain')
    if not chain:
        return jsonify({'error': 'Pipeline not initialized. Please upload a PDF first.'}), 400
    
    try:
        answer = ask_question(chain, question)
        return jsonify({'answer': answer}), 200
    except Exception as e:
        print(f"Error asking question: {e}") # Log the error
        return jsonify({'error': f'Failed to get answer: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(host='0.0.0.0', port=8000, debug=True)
