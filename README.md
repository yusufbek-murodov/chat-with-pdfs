# Chat with PDF using Gemini

## Overview
This project allows users to interact with PDF files using a conversational interface powered by Google's Gemini model and LangChain. Users can upload PDFs, process their content into vector embeddings, and ask questions about the content, receiving detailed and context-aware answers.

## Features
- Extract text from uploaded PDFs.
- Split text into manageable chunks for processing.
- Generate embeddings using Google's Generative AI.
- Store and retrieve embeddings using FAISS for similarity search.
- Chat interface for question-answering based on PDF content.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web interface for user interaction.
- **PyPDF2**: Extract text from PDF files.
- **LangChain**: Text processing and conversational AI framework.
- **Google Generative AI**: Embedding and chat model.
- **FAISS**: Vector store for efficient similarity search.
- **dotenv**: Manage environment variables.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.9 or later
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file:
   - Create a `.env` file in the root directory.
   - Add your Google Generative AI API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```

## How to Run
1. Launch the application:
   ```bash
   streamlit run main.py
   ```
2. Open the URL provided in the terminal to access the app.
3. Use the sidebar to upload PDF files.
4. Ask questions about the content in the main interface.

Alternatively, you can try the deployed version of this app [here](https://chat-with-pdfs-yusufbek.streamlit.app/).

## Project Structure
```
|-- main.py                  # Entry point of the application
|-- requirements.txt         # Python dependencies
|-- .env                     # Environment variables (ignored in version control)
|-- README.md                # Project documentation
```

## Usage
### Uploading PDFs
- Use the sidebar to upload one or multiple PDF files.
- Click "Submit & Process" to extract text and store embeddings.

### Asking Questions
- Enter a question in the text input box on the main screen.
- View the AI's response below the input box.

## Key Functions

### `get_pdf_text(pdf_docs)`
Extracts text from the uploaded PDF files.
- **Parameters**:
  - `pdf_docs`: List of uploaded PDF files.
- **Returns**: Extracted text as a single string.

### `get_text_chunks(text)`
Splits the text into smaller, overlapping chunks for processing.
- **Parameters**:
  - `text`: Raw text extracted from PDFs.
- **Returns**: List of text chunks.

### `get_vector_store(text_chunks, save=False)`
Creates a vector store using embeddings and optionally saves it locally.
- **Parameters**:
  - `text_chunks`: List of text chunks to embed.
  - `save`: Boolean flag to save the vector store locally.

### `get_conversational_chain()`
Configures the question-answering chain using Gemini.
- **Returns**: Configured LangChain object.

### `user_input(user_question)`
Handles user questions and provides responses.
- **Parameters**:
  - `user_question`: Question asked by the user.

### `main()`
Sets up the Streamlit interface and handles user interactions.


## License
This project is licensed under the MIT License. See `LICENSE` for details.

