# ClauseCompass 🧭 - HackRx 6.0 Submission

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![AI/LLM](https://img.shields.io/badge/AI_/_LLM-Google_Gemini_1.5-purple.svg)](https://deepmind.google/technologies/gemini/)

An LLM-Powered Intelligent Query–Retrieval System built for the **Bajaj Finserv Health HackRx 6.0** hackathon. ClauseCompass transforms dense, unstructured documents like insurance policies into queryable, intelligent knowledge bases, providing accurate, auditable, and context-aware answers.

<!-- 
**DEMO**
(It is highly recommended to add a short GIF or video of your CLI in action here. 
This is the most impactful part of a README!)
![Demo of ClauseCompass CLI](link_to_your_demo.gif)
-->

## 🚀 The Challenge

The hackathon's problem statement was to build a system that could process natural language queries against large, unstructured documents and make contextual decisions. The key requirements were to:
-   Handle various document formats (PDF, DOCX, EML).
-   Use semantic search, not just keyword matching.
-   Provide explainable, auditable answers.
-   Return structured responses for downstream use.

## ✨ Our Solution: ClauseCompass

ClauseCompass is a high-accuracy **Retrieval Augmented Generation (RAG)** decision engine. Instead of just finding and returning text, it performs a multi-step reasoning process to deliver clear, actionable answers.

Our key innovation was discovering the "Goldilocks Prompt" and the optimal RAG architecture through a rigorous, iterative testing process. We proved that by leveraging a powerful model like **Google's Gemini 1.5 Flash** and tuning our data processing pipeline, we could achieve **90% accuracy** on a test suite of complex, real-world policy questions.

### Key Features

*   **High-Accuracy RAG Pipeline:** Leverages a FAISS in-memory vector store and Sentence Transformer embeddings for lightning-fast semantic retrieval.
*   **Advanced Reasoning with Google Gemini 1.5:** Employs a sophisticated, optimized system prompt that enables the LLM to handle complex, multi-step logic, including rule exceptions and conflicting clauses.
*   **Flexible Data Ingestion:** Uses the `unstructured` library to seamlessly process and extract text from multiple document formats, including PDF, Word (DOCX), and emails (EML).
*   **Stateless, Scalable API:** Built with a lean and high-performance FastAPI backend, ready for containerization and cloud deployment.
*   **Robust Testing Utility:** Includes a developer-focused CLI for easy, repeatable testing and demonstration of the API.

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **AI / LLM:** Google Gemini 1.5 Flash
*   **Vector Store:** FAISS (in-memory)
*   **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
*   **Data Processing:** LangChain (Text Splitters), Unstructured.io (Document Loaders)
*   **Deployment:** Docker, Coolify, Google Cloud Platform (GCP)
*   **CLI / Testing:** Typer, Rich, Requests

## ⚙️ Setup and Deployment

This application is designed to be deployed as a Docker container, managed by a self-hosted PaaS like Coolify running on a cloud VM.

### Prerequisites

*   A Google Cloud Platform (or any cloud provider) account.
*   A VM instance (e.g., GCP Compute Engine `n2-standard-2` or higher) running Debian 12.
*   A `.env` file with your `GOOGLE_API_KEY`.
*   Docker installed on the VM.
*   Coolify installed on the VM.

### Deployment Steps

1.  **Launch and Configure a Cloud VM:**
    *   Create a VM instance on GCP/AWS (Debian 12 is recommended).
    *   Ensure firewall rules allow inbound traffic on ports `22` (SSH), `80` (HTTP), `443` (HTTPS), and `3000` (Coolify UI).
    *   Reserve a static IP address for the VM.

2.  **Install Coolify:**
    *   SSH into your VM and install Docker and then Coolify using the official installation script from `get.coolify.io`.

3.  **Deploy via Coolify:**
    *   Log in to your Coolify dashboard.
    *   Connect your GitHub account as a "Source".
    *   Create a new "Project" and add a "Resource" pointing to this GitHub repository.
    *   **Set the Build Pack to `Dockerfile`** and the **Port to `8000`**.
    *   In the **Domain (FQDN)** field, use a `sslip.io` address or a custom domain pointing to your server's static IP (e.g., `https://clausecompass.[YOUR_IP].sslip.io`).
    *   Add your `GOOGLE_API_KEY` in the **Secrets** tab.
    *   Click **Deploy**. Coolify will automatically build the Docker image, run the container, configure the reverse proxy, and provision an SSL certificate.

### Local Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nightmare-tech/clausecompass-hackrx.git
    cd clausecompass-hackrx
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure `.env` file:**
    Create a `.env` file and add your `GOOGLE_API_KEY`.
4.  **Run the Server:**
    ```bash
    python app.py
    ```
5.  **Run the CLI Client (in a separate terminal):**
    ```bash
    # For local testing
    python cli.py

    # To test the deployed server, set the environment variable first:
    export HACKRX_API_URL="https://your-deployed-url.com"
    python cli.py
    ```

## 📈 Key Learnings & Future Enhancements

Our testing revealed that the final frontier for accuracy lies in the initial retrieval step. In rare cases, a pure semantic search can fail to retrieve a specific key term (like 'Hernia') when buried in a long list.

To solve this and reach near-100% accuracy, our primary future enhancement would be to implement **Hybrid Search**:
*   Combine our current FAISS-based semantic search with a traditional keyword search algorithm (like BM25).
*   The results would be re-ranked to provide the LLM with the most relevant possible context, guaranteeing that critical keywords are never missed.

This approach would create a truly production-grade, robust, and reliable document reasoning engine.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

![arch-data-flow](RealTimeQueryFlow.png)
