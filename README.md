
# Email Agent – Coding Assessment

## Overview

This project implements an **Email Agent** powered by **GPT-5**.
The agent accepts two inputs:

1. A **text prompt**
2. An **email file**

The behavior of the agent depends on the content of the text prompt:

* **Similarity-related prompts**
  → The agent performs a tool call to `email_similarity_search`.
  → The tool uses **ChromaDB** as the vector store to embed, store, and query email data.

* **Non-similarity prompts**
  → The agent does not perform any tool calls. Instead, it returns an **error message** stating that no tool calls have been added for this type of request.

This setup demonstrates prompt-based tool routing, vector search, and error handling.

---

## How It Works

1. **Input**

   * User provides a text query and an email file.

2. **Embedding & Storage**

   * Emails are processed into embeddings using **ChromaDB**.
   * All embeddings are stored in the vector database for fast retrieval.

3. **Similarity Search**

   * If the prompt indicates similarity intent, the agent queries ChromaDB.
   * ChromaDB returns the closest-matching emails.

4. **Routing Logic**

   * If similarity intent → call `email_similarity_search`.
   * Otherwise → return error response.

5. **Output**

   * Similarity case → returns relevant email(s).
   * Non-similarity case → returns error message.

---

## Screenshots

* ✅ **Similarity Prompt (Tool Call Triggered)**
<img width="487" height="28" alt="Screenshot 2025-10-01 at 1 28 14 PM" src="https://github.com/user-attachments/assets/116dd132-3f5e-4be4-91e6-f571910dea92" />

* 🗂️ **ChromaDB Processing Embeddings**
<img width="782" height="703" alt="Screenshot 2025-10-01 at 1 29 27 PM" src="https://github.com/user-attachments/assets/69fb050d-0e3e-4de6-b7f7-deeb06d5dbe6" />
<img width="636" height="656" alt="Screenshot 2025-10-01 at 1 31 42 PM" src="https://github.com/user-attachments/assets/b386c1be-cb94-45c8-93b3-2c93e9a50b2a" />
<img width="690" height="773" alt="Screenshot 2025-10-01 at 1 56 17 PM" src="https://github.com/user-attachments/assets/bab704e9-597a-4e2c-ba81-dc9bae9c48fd" />

<img width="662" height="744" alt="Screenshot 2025-10-01 at 1 56 06 PM" src="https://github.com/user-attachments/assets/79f34f48-bfdf-4fdc-a228-a79f3ffcf31d" />


* 📬 **ChromaDB Similar Email Output**
<img width="683" height="320" alt="Screenshot 2025-10-01 at 1 58 29 PM" src="https://github.com/user-attachments/assets/02a8364e-a60e-48b1-b7b1-f99875bb949f" />

<img width="754" height="459" alt="Screenshot 2025-10-01 at 1 57 52 PM" src="https://github.com/user-attachments/assets/6f310774-211d-4cfb-b6b2-8722416dcaaf" />


---

## Tech Stack

* **Model**: GPT-5 (primary reasoning agent)
* **Vector Store**: ChromaDB (for embeddings and similarity search)
* **Tools**: `email_similarity_search`


Do you want me to make this **submission-polished** (with crisp Markdown formatting and GitHub-style code snippets for usage examples), or keep it **shorter and reviewer-friendly**?
