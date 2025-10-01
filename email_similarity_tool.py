"""
Email Similarity Search Tool for Agentic System
This module provides the email similarity search functionality as a tool call
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from email import message_from_file
from email.mime.text import MIMEText
import openai
from pathlib import Path
import logging
from dataclasses import dataclass
from functools import lru_cache
import re
import html
from bs4 import BeautifulSoup
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmailData:
    """Data class to hold email information and embeddings"""
    filename: str
    subject: str
    sender: str
    content: str
    embedding: Optional[List[float]] = None

class EmailSimilarityTool:
    """Tool class for email similarity search that can be called by an agent"""
    
    def __init__(self, api_key: str, email_dir: str = "email"):
        """
        Initialize the email similarity search tool
        
        Args:
            api_key: OpenAI API key
            email_dir: Directory containing .eml files
        """
        self.api_key = api_key
        self.email_dir = Path(email_dir)
        self.chroma_db_path = self.email_dir / "chroma_db"
        self.emails_data: Dict[str, EmailData] = {}
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        self.collection_name = "email_embeddings"
        
        # Load or create embeddings in ChromaDB
        self._load_or_create_embeddings()
    
    def _extract_email_content(self, eml_file_path: Path) -> EmailData:
        """Extract meaningful content from an .eml file with improved preprocessing"""
        try:
            with open(eml_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = message_from_file(f)
            
            # Extract and clean subject
            subject = self._extract_and_clean_subject(msg)
            
            # Extract sender
            sender = msg.get('From', '')
            
            # Extract and clean email body content
            content = self._extract_and_clean_body(msg)
            
            return EmailData(
                filename=eml_file_path.name,
                subject=subject,
                sender=sender,
                content=content
            )
            
        except Exception as e:
            logger.error(f"Error processing {eml_file_path}: {e}")
            return EmailData(
                filename=eml_file_path.name,
                subject="",
                sender="",
                content=""
            )
    
    def _extract_and_clean_subject(self, msg) -> str:
        """Extract and clean email subject line"""
        subject = msg.get('Subject', '')
        if not subject:
            return ""
        
        # Decode MIME encoded words (RFC 2047)
        try:
            from email.header import decode_header
            decoded_parts = decode_header(subject)
            decoded_subject = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_subject += part.decode(encoding, errors='ignore')
                    else:
                        decoded_subject += part.decode('utf-8', errors='ignore')
                else:
                    decoded_subject += part
            subject = decoded_subject
        except Exception:
            # Fallback to simple decoding
            try:
                subject = subject.encode('utf-8').decode('unicode_escape')
            except:
                pass
        
        # Clean up the subject
        subject = re.sub(r'\s+', ' ', subject.strip())
        return subject
    
    def _extract_and_clean_body(self, msg) -> str:
        """Extract and clean email body content, prioritizing text over HTML"""
        content = ""
        
        if msg.is_multipart():
            # Look for text/plain first, then text/html
            text_content = ""
            html_content = ""
            
            for part in msg.walk():
                content_type = part.get_content_type()
                payload = part.get_payload(decode=True)
                
                if payload:
                    try:
                        decoded_payload = payload.decode('utf-8', errors='ignore')
                        
                        if content_type == "text/plain":
                            text_content = decoded_payload
                        elif content_type == "text/html" and not text_content:
                            html_content = decoded_payload
                    except Exception:
                        continue
            
            # Prefer plain text, fallback to HTML
            if text_content:
                content = self._clean_text_content(text_content)
            elif html_content:
                content = self._clean_html_content(html_content)
        else:
            # Single part message
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    decoded_payload = payload.decode('utf-8', errors='ignore')
                    content_type = msg.get_content_type()
                    
                    if content_type == "text/plain":
                        content = self._clean_text_content(decoded_payload)
                    elif content_type == "text/html":
                        content = self._clean_html_content(decoded_payload)
                    else:
                        content = self._clean_text_content(decoded_payload)
                except Exception:
                    pass
        
        # Final cleanup and length limiting
        content = re.sub(r'\s+', ' ', content.strip())
        if len(content) > 4000:  # Limit content length for API efficiency
            content = content[:4000]
        
        return content
    
    def _clean_text_content(self, text: str) -> str:
        """Clean plain text content"""
        # Remove quoted-printable encoding artifacts
        text = re.sub(r'=\r?\n', '', text)
        text = re.sub(r'=[0-9A-F]{2}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common email artifacts
        text = re.sub(r'--\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and extract readable text"""
        try:
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up the extracted text
            text = self._clean_text_content(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error parsing HTML content: {e}")
            # Fallback to simple regex-based HTML tag removal
            text = re.sub(r'<[^>]+>', '', html_content)
            return self._clean_text_content(text)
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings from ChromaDB or create new ones"""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection with {self.collection.count()} emails")
            
            # Load email metadata for compatibility
            self._load_email_metadata()
            
            # Check for new emails and add them
            self._add_new_emails_to_collection()
        except Exception:
            logger.info("No existing collection found, creating new embeddings...")
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for all emails in the directory and store in ChromaDB"""
        eml_files = list(self.email_dir.glob("*.eml"))
        logger.info(f"Found {len(eml_files)} .eml files")
        
        # Create new collection
        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        
        # Prepare data for batch insertion
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, eml_file in enumerate(eml_files):
            print(f"Processing {i+1}/{len(eml_files)}: {eml_file.name}")
            
            # Extract email content
            email_data = self._extract_email_content(eml_file)
            
            # Create embedding for the content
            if email_data.content:
                # Combine subject and content for better semantic understanding
                combined_text = f"{email_data.subject} {email_data.content}"
                email_data.embedding = self._get_openai_embedding(combined_text)
                
                # Only add to ChromaDB if embedding was successfully created
                if email_data.embedding and len(email_data.embedding) > 0:
                    ids.append(eml_file.name)
                    embeddings.append(email_data.embedding)
                    metadatas.append({
                        "filename": email_data.filename,
                        "subject": email_data.subject,
                        "sender": email_data.sender,
                        "content_length": len(email_data.content)
                    })
                    documents.append(combined_text)
                else:
                    logger.warning(f"Skipping {eml_file.name} - failed to generate embedding")
            
            self.emails_data[eml_file.name] = email_data
        
        # Batch insert into ChromaDB
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Successfully stored {len(ids)} email embeddings in ChromaDB")
        else:
            logger.warning("No emails with content found to store")
        
        logger.info("Embeddings created and stored in ChromaDB successfully")
    
    def _load_email_metadata(self):
        """Load email metadata from ChromaDB for compatibility with existing interface"""
        try:
            # Get all documents from ChromaDB to populate emails_data
            results = self.collection.get()
            
            if results and 'ids' in results and results['ids']:
                for i, filename in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if 'metadatas' in results and results['metadatas'] else {}
                    embedding = results['embeddings'][i] if 'embeddings' in results and results['embeddings'] else None
                    document = results['documents'][i] if 'documents' in results and results['documents'] else ""
                    
                    # Reconstruct EmailData object
                    email_data = EmailData(
                        filename=metadata.get('filename', filename),
                        subject=metadata.get('subject', ''),
                        sender=metadata.get('sender', ''),
                        content=document,  # This is the combined text
                        embedding=embedding
                    )
                    
                    self.emails_data[filename] = email_data
                
                logger.info(f"Loaded metadata for {len(self.emails_data)} emails from ChromaDB")
            else:
                logger.warning("No data found in ChromaDB collection")
                self.emails_data = {}
        except Exception as e:
            logger.error(f"Error loading email metadata from ChromaDB: {e}")
            self.emails_data = {}
    
    def _add_new_emails_to_collection(self):
        """Check for new emails in the directory and add them to ChromaDB"""
        try:
            # Get all email files in the directory
            all_eml_files = set(eml_file.name for eml_file in self.email_dir.glob("*.eml"))
            # Get existing emails in ChromaDB
            existing_emails = set(self.emails_data.keys())
            
            # Find new emails
            new_emails = all_eml_files - existing_emails
            
            if new_emails:
                logger.info(f"Found {len(new_emails)} new emails to add to ChromaDB")
                
                # Prepare data for batch insertion
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for i, filename in enumerate(sorted(new_emails)):
                    eml_file = self.email_dir / filename
                    print(f"Processing {i+1}/{len(new_emails)}: {filename}")
                    
                    # Extract email content
                    email_data = self._extract_email_content(eml_file)
                    
                    # Create embedding for the content
                    if email_data.content:
                        combined_text = f"{email_data.subject} {email_data.content}"
                        email_data.embedding = self._get_openai_embedding(combined_text)
                        
                        # Only add to ChromaDB if embedding was successfully created
                        if email_data.embedding and len(email_data.embedding) > 0:
                            ids.append(filename)
                            embeddings.append(email_data.embedding)
                            metadatas.append({
                                "filename": email_data.filename,
                                "subject": email_data.subject,
                                "sender": email_data.sender,
                                "content_length": len(email_data.content)
                            })
                            documents.append(combined_text)
                        else:
                            logger.warning(f"Skipping {filename} - failed to generate embedding")
                    
                    self.emails_data[filename] = email_data
                
                # Batch insert into ChromaDB
                if ids:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents
                    )
                    logger.info(f"Successfully added {len(ids)} new email embeddings to ChromaDB")
                    logger.info(f"ChromaDB collection now contains {self.collection.count()} emails")
            else:
                logger.info("No new emails to add - collection is up to date")
        except Exception as e:
            logger.error(f"Error adding new emails to ChromaDB: {e}")
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings (kept for compatibility)"""
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarityt
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_emails(self, email_file: str, top_k: int = 10, min_similarity: float = 0.0) -> Dict:
        """
        Find similar emails to the given email file using ChromaDB - TOOL CALL INTERFACE
        
        Args:
            email_file: Path to the input email file
            top_k: Number of similar emails to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dictionary with results for agent consumption
        """
        email_path = Path(email_file)
        
        # Extract content from input email
        input_email = self._extract_email_content(email_path)
        
        if not input_email.content:
            return {
                "success": False,
                "error": f"No content found in {email_file}",
                "input_email": {
                    "filename": input_email.filename,
                    "subject": input_email.subject,
                    "sender": input_email.sender,
                    "content_length": len(input_email.content)
                },
                "similar_emails": []
            }
        
        # Get embedding for input email
        combined_text = f"{input_email.subject} {input_email.content}"
        input_embedding = self._get_openai_embedding(combined_text)
        
        if not input_embedding:
            return {
                "success": False,
                "error": f"Failed to get embedding for {email_file}",
                "input_email": {
                    "filename": input_email.filename,
                    "subject": input_email.subject,
                    "sender": input_email.sender,
                    "content_length": len(input_email.content)
                },
                "similar_emails": []
            }
        
        try:
            # Query ChromaDB for similar emails
            print(f"🔍 Querying ChromaDB for {top_k} similar emails...")
            results = self.collection.query(
                query_embeddings=[input_embedding],
                n_results=top_k + 1,  # +1 to account for potential self-match
                where={"filename": {"$ne": email_path.name}}  # Exclude the input email itself
            )
            print(f"✓ ChromaDB returned {len(results['ids'][0]) if results['ids'] else 0} results")
            
            # Process results
            similar_emails = []
            total_found = 0
            
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]
                
                # Convert distances to similarity scores (ChromaDB uses cosine distance)
                # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance
                similarities = [1 - distance for distance in distances]
                
                # Filter by minimum similarity threshold
                filtered_results = []
                for i, similarity in enumerate(similarities):
                    if similarity >= min_similarity:
                        filtered_results.append((ids[i], similarity, metadatas[i], documents[i]))
                        total_found += 1
                
                # Sort by similarity (descending)
                filtered_results.sort(key=lambda x: x[1], reverse=True)
                
                # Format results for agent consumption
                for filename, score, metadata, document in filtered_results[:top_k]:
                    similar_emails.append({
                        "filename": filename,
                        "similarity_score": round(score, 4),
                        "subject": metadata['subject'],
                        "sender": metadata['sender'],
                        "content_length": metadata['content_length'],
                        "content_preview": document[:200] + "..." if len(document) > 200 else document
                    })
            
            return {
                "success": True,
                "input_email": {
                    "filename": input_email.filename,
                    "subject": input_email.subject,
                    "sender": input_email.sender,
                    "content_length": len(input_email.content),
                    "content_preview": input_email.content[:200] + "..." if len(input_email.content) > 200 else input_email.content
                },
                "similar_emails": similar_emails,
                "total_found": total_found,
                "returned": len(similar_emails),
                "min_similarity_threshold": min_similarity
            }
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return {
                "success": False,
                "error": f"ChromaDB query failed: {str(e)}",
                "input_email": {
                    "filename": input_email.filename,
                    "subject": input_email.subject,
                    "sender": input_email.sender,
                    "content_length": len(input_email.content)
                },
                "similar_emails": []
            }

# Global tool instance
_email_similarity_tool = None

def initialize_email_similarity_tool(api_key: str, email_dir: str = "email") -> EmailSimilarityTool:
    """Initialize the global email similarity tool"""
    global _email_similarity_tool
    _email_similarity_tool = EmailSimilarityTool(api_key, email_dir)
    return _email_similarity_tool

def email_similarity_search(email_file: str, top_k: int = 10, min_similarity: float = 0.0) -> Dict:
    """
    Tool call function for email similarity search
    
    Args:
        email_file: Path to the input email file
        top_k: Number of similar emails to return (default: 10)
        min_similarity: Minimum similarity threshold (default: 0.0)
        
    Returns:
        Dictionary with search results
    """
    global _email_similarity_tool
    if _email_similarity_tool is None:
        raise RuntimeError("Email similarity tool not initialized. Call initialize_email_similarity_tool() first.")
    
    return _email_similarity_tool.find_similar_emails(email_file, top_k, min_similarity)
