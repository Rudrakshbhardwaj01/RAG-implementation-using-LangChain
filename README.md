# RAG-Based PDF Workflows

Welcome! This repository features **two distinct Python workflows** for working with PDFs using Retrieval-Augmented Generation (RAG) and local LLMs. Both run entirely on your machine—**no API tokens required**. You provide your own PDFs, and each PDF gets its own vector store for efficient processing.

---

## Table of Contents

- [Overview](#overview)
- [PDF Summarizer](#pdf-summarizer)
- [PDF Assist (PDF Q&A)](#pdf-assist-pdf-qa)
- [Requirements](#requirements)
- [Usage](#usage)
- [Notes & Tips](#notes--tips)

---

## Overview

This repository provides **two local RAG systems** for PDF analytics:

1. **PDF Summarizer (`pdfSummarizer.py`)**  
   - Generates a concise, 3–5 sentence summary for any PDF.

2. **PDF Assist (`pdfAssist.py`)**  
   - Lets you interactively ask any question about your PDF and get accurate, context-based answers.

> **Key Points:**  
> - You must supply your own PDFs.  
> - Each PDF needs its own vector store (database)—set a unique `persist_directory` name for each PDF.
> - 100% local: **No API tokens or external calls needed.**

---

## PDF Summarizer

**Purpose:**  
Produce a short, clear summary of your PDF using a local RAG workflow.

**How it works:**
1. **Load your PDF:**  
   - Uses `PyPDFLoader` to extract text from the PDF file you provide.
2. **Split text into chunks:**  
   - Uses `RecursiveCharacterTextSplitter` for optimal chunking.
3. **Generate embeddings:**  
   - Employs HuggingFace embedding models locally.
4. **Store embeddings:**  
   - Embeddings are stored in a Chroma vector store (database) using your chosen `persist_directory` name (ensure it's unique per PDF).
5. **Generate summary:**  
   - Retrieves relevant content and summarizes using a local LLM (example: Flan-T5 from HuggingFace).

**Example Usage:**
```bash
python pdfSummarizer.py
```
- **Instructions:**  
  - Update the script to use the correct PDF file path.
  - The concise summary will be printed to your terminal.

---

## PDF Assist (PDF Q&A)

**Purpose:**  
Interactively ask any question about your PDF and receive answers based on the document content.

**How it works:**
1. **Provide your PDF:**  
   - Enter the correct file path for your PDF.
2. **Load and chunk:**  
   - Uses `PyPDFLoader` and splits text into chunks.
3. **Generate and store embeddings:**  
   - Chunks are embedded and saved to a Chroma vector store.
   - **Important:** Specify a unique `persist_directory` for each PDF to keep databases separate.
4. **Ask questions:**  
   - Query the PDF's content interactively.  
   - Answers generated via a local LLM (example: TinyLlama).

**Example Usage:**
```bash
python pdfAssist.py
```
- **Instructions:**  
  - Be sure to specify the correct PDF path and a unique vector store name for each PDF.
  - Enter questions interactively; answers are printed to your terminal.

---

## Requirements

All required libraries are listed in `requirements.txt`.  
Simply install them with:

```bash
pip install -r requirements.txt
```

- No API tokens or cloud dependencies required.
- Everything runs locally.

---

## Usage

**PDF Summarizer:**
1. Place your PDF in an accessible location.
2. Update `pdfSummarizer.py` with your PDF's file path.
3. Run:
   ```bash
   python pdfSummarizer.py
   ```
4. View the summary in your terminal.

**PDF Assist (PDF Q&A):**
1. Place your PDF in an accessible location.
2. Update `pdfAssist.py` with your PDF path and a unique vector store (`persist_directory`) name.
3. Run:
   ```bash
   python pdfAssist.py
   ```
4. Ask questions interactively.

---

## Notes & Tips

- **Chunk size & retrieval (`k`):**  
  - Adjust chunk size and retrieval `k` in the scripts for best results with different PDFs.
- **Memory:**  
  - Large LLMs require more RAM and may benefit from GPU acceleration.
- **Vector store reuse:**  
  - For repeated use of the same PDF, reuse its vector store to avoid recomputing embeddings.
- **Scanned PDFs:**  
  - Require OCR. Text-based PDFs work out-of-the-box.
- **Unique vector stores:**  
  - Always use a unique `persist_directory` for each PDF.
- **Troubleshooting:**  
  - If results seem off, check your chunk size, PDF format, and vector store settings.

---

## License

This project is licensed under the MIT License.

---

**Happy Summarizing and Q&A!**
