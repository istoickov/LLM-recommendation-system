# LLM Recommendation System

## Overview

This system uses **Large Language Models (LLMs)** to generate personalized recommendations by querying a database of profiles. The backend is built with **FastAPI** and integrates with **FAISS** for efficient semantic search and **Redis** for caching recent queries. The frontend, built with **React**, provides an interactive interface for submitting queries and viewing recommendations.

## Features

- **Backend**:
  - FastAPI-based REST API for searching profiles.
  - FAISS-based indexing for efficient similarity search.
  - Redis caching for storing recent queries and their results.

- **Frontend**:
  - Query input field.
  - Dropdowns for selecting model and preprocessing type.
  - Displays the closest 10 profiles and reusable query combinations.

## Installation

### Prerequisites

- Python 3.8+
- Node.js and npm

### Backend Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/istoickov/LLM-recommendation-system.git
    cd LLM-recommendation-system
    ```

2. Install backend dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3. Configure Redis and FAISS if needed.

4. Run the FastAPI app:
    ```bash
    uvicorn main:app --reload
    ```

### Frontend Setup

1. Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2. Install frontend dependencies:
    ```bash
    npm install
    ```

3. Run the React app:
    ```bash
    npm start
    ```

4. Open the app in your browser at `http://localhost:3000`.

## Usage

1. Enter a query into the input field.
2. Select the model and preprocessing type from the dropdown menus.
3. Click the submit button to receive the top 10 closest profiles.
4. The combination of the query, model, and preprocessing type will be displayed on the left side for future use.

![Demo](demo.webp)

## Configuration

- **Backend**:
  - Modify `config.py` for Redis and FAISS settings.
  - Choose different LLMs by updating the `llms.py` file.

## TODO

- Add direnv (.env, .env.local)
- Add unit and integration tests
- Add CI/CD pipeline
- Deploy
- Add pre-commit (black, isort, etc.)
- Code coverage checks
- Fix frontend UI for displaying the last 10 used queries.
- Refactor backend code for better maintainability.
- Add real FAISS indices with dummy data.
- Add evaluation metrics after masking the original data with dummy data.
- Improve frontend UI for better user experience.
- Try training fine-tuned models with more than 1 epoch (currently limited).

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the **AGPL-3.0 License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **FastAPI**: For building a fast and efficient API.
- **FAISS**: For high-performance similarity search.
- **Redis**: For caching query results.
- **React**: For building the interactive frontend interface.
