"""
RAG Pipeline – Uses WebBaseLoader to parse photography tips and LLM to suggest
dynamic, contextual improvements based on extracted FrameSense features.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("framesense.rag")


class RAGPipeline:
    def __init__(self, urls: list[str] = None):
        """
        Initialize the RAG pipeline by loading documents, creating the vector store,
        and setting up the LLM chain.
        """
        self.urls = urls or [
            "https://expertphotography.com/rule-of-thirds/",
            "https://en.wikipedia.org/wiki/Rule_of_thirds",
            "https://digital-photography-school.com/rule-of-thirds/"
        ]
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.vector_store = None
        self.llm = None
        self.retriever = None
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. RAG pipeline is disabled.")
            return
            
        try:
            self._setup_rag()
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            self.llm = None

    def _setup_rag(self):
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        
        if os.path.exists(index_path):
            logger.info("Loading local FAISS index...")
            self.vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Initializing RAG WebBaseLoader...")
            # Use headers and a timeout to prevent the scraper from being blocked or hanging indefinitely
            loader = WebBaseLoader(
                self.urls,
                header_template={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5"
                },
                requests_kwargs={"timeout": 10}
            )
            
            try:
                # aload() runs asyncio gathering for much faster concurrent fetching
                docs = loader.aload()
            except Exception as e:
                logger.warning(f"aload() failed ({e}), falling back to sequential load().")
                docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            self.vector_store = FAISS.from_documents(splits, embeddings)
            self.vector_store.save_local(index_path)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key, temperature=0.2)
        
        self.prompt = PromptTemplate.from_template(
            "You are an expert photography assistant. Use the following pieces of retrieved "
            "photography context to provide 3-4 short, actionable suggestions for a user taking a photo.\n"
            "Keep the suggestions very brief (like 'Move left', 'Increase brightness', 'Step back', 'Rotate camera with angle mention,and also mention exposure settings').\n\n"
            "Context: {context}\n\n"
            "Current Scene Features:\n"
            "Brightness: {brightness}\n"
            "Focus: {focus}\n"
            "Clutter: {clutter}\n"
            "Alignment: {alignment}\n"
            "Depth: {depth}\n"
            "Subject: {subject_detected} ({subject_position})\n\n"
            "Suggestions:"
        )

    def generate_suggestion(self, features: dict[str, Any]) -> list[str]:
        """
        Given the extracted features, use RAG to generate suggestions.
        """
        if not self.llm:
            return []
            
        try:
            # Create a simple summary query
            query = f"Photography advice for {features.get('brightness', 'normal')} lighting, {features.get('alignment', 'normal')} alignment."
            logger.info(f"RAG: retrieving context for query: {query}")
            
            retrieved_docs = self.retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            logger.info(f"RAG: retrieved {len(retrieved_docs)} documents.")
            
            prompt_val = self.prompt.format(
                context=context,
                brightness=features.get("brightness", "unknown"),
                focus=features.get("focus", "unknown"),
                clutter=features.get("background_clutter", "unknown"),
                alignment=features.get("alignment", "unknown"),
                depth=features.get("distance", "unknown"),
                subject_detected=features.get("subject_detected", False),
                subject_position=features.get("subject_position", "unknown")
            )
            
            logger.info("RAG: invoking LLM...")
            response = self.llm.invoke(prompt_val)
            raw_text = response.content.strip()
            logger.info(f"RAG: LLM response received ({len(raw_text)} chars).")
            
            suggestions = [line.strip("- *").strip() for line in raw_text.split("\n") if line.strip()]
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating RAG suggestions: {e}", exc_info=True)
            return []
