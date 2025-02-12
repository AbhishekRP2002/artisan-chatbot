# Artisan RAG Chatbot

This FastAPI project provides a chatbot interface and a deployed API with a focus on providing information about Artisan and Ava, leveraging a Retrieval-Augmented Generation (RAG) approach.  Deployed on [Render](render.com)

## Objective

To build a FastAPI app with a `/chat` endpoint that takes a user message and responds appropriately, considering chat history and a knowledge base scraped from [https://artisan.co](https://artisan.co).

## Key Features

* **Chat History:** The chatbot considers the last 10 messages in the conversation as conversational buffer memory to provide contextually relevant responses.
* **Knowledge Base:**  Information about Artisan platform and Ava, AI SDR is scraped from multiple web pages within the Artisan website and used as a knowledge base for generating responses.
* **RAG Implementation:** The application uses a Retrieval-Augmented Generation (RAG) approach to combine retrieved knowledge with LLM-based generation for accurate and informative responses.
* **Deployed on Render.com:** The API is deployed and accessible via Render.com. There is also a chatbot interface providing the UI for the conversational support chatbot.
  * Deployed API service : https://artisan-chatbot.onrender.com/docs
  * Deployed chatbot interface : https://artisan-chatbot.onrender.com/chainlit/

## Approach

- Data ingestion in the Vector DB
  - Identify the list of web pages whose content will be used as the knowledge base
  - Scrape the contents of the selected web pages using Firecrawl.
  - Clean the scraped raw markdown contents to remove the noise and reduce the overall number of chunks ( reduction of search space )
  - Chunk each of the parent document using a combination of Semantic Chunking + Recursive Character based chunking which takes into account both contextual meaning and document structure.
  - Embed and store in the Milvus vector DB as a collection with each chunk as a record.
- Create a Conversational RAG pipeline using LangChain
- Create a FastAPI app exposing the `/chat` endpoint which acts as the entry point for our RAG system
- To create the Conversational Chatbot UI use Chainlit mouted as a sub-application in FastAPI.

## API Endpoint

* `/chat`:  This endpoint accepts a POST request with a JSON payload containing the user's input message. It returns a JSON response containing the chatbot's reply.

  **Curl Request:**

  ```json
  curl -X POST "https://artisan-chatbot.onrender.com/chat" \
       -H "Content-Type: application/json" \
       -d '{
             "message": "Hello, how can you help me?",
             "session_id": "12345"
           }'

  ```

  **Response:**

  ```json
  {"response":"Hello. I'm here to assist you with any questions or concerns you may have about Artisan's AI-powered platform and its products, such as Ava, the AI Sales Agent. I can provide information on features like Email Warmup, Sales Automation, Email Personalization, LinkedIn Outreach, and our comprehensive B2B data. If you're experiencing issues or need guidance on setup, pricing, integrations, or best practices, I'm here to help. What specifically would you like to know or discuss about Artisan's solutions?"}
  ```

## Deployment

This application is deployed on [Render.com](https://render.com).
