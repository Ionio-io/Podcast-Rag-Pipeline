import os
from openai import OpenAI
from typing import List, Dict, Any
import time

class ResponsesAPIRAG:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.vector_store = None
        self.uploaded_files = []
        self.conversation_history = []
        
    def create_vector_store(self, name: str = "Transcript Documents"):
        """Create a vector store for documents."""
        try:
            self.vector_store = self.client.vector_stores.create(name=name)
            print(f"Created vector store: {self.vector_store.id}")
            return self.vector_store
        except AttributeError:
            print("Vector stores not available in current OpenAI client version")
            raise ValueError("Vector stores API not available. Please upgrade OpenAI client or use a different approach.")
        
    def upload_file(self, file_path: str) -> str:
        """Upload a single file to OpenAI and add to vector store."""
        try:
            # Upload file to OpenAI with explicit UTF-8 encoding
            with open(file_path, 'rb') as f:
                file = self.client.files.create(file=f, purpose="assistants")
            
            # Add file to vector store
            self.client.vector_stores.files.create(
                vector_store_id=self.vector_store.id,
                file_id=file.id
            )
            
            # Ensure filename is properly encoded
            filename = os.path.basename(file_path)
            # Handle any potential Unicode issues in filename
            try:
                filename_safe = filename.encode('utf-8').decode('utf-8')
            except UnicodeError:
                filename_safe = filename.encode('ascii', errors='ignore').decode('ascii')
            
            self.uploaded_files.append({
                "file_id": file.id,
                "filename": filename_safe,
                "file_path": file_path
            })
            
            print(f"Uploaded: {filename_safe} (ID: {file.id})")
            return file.id
        except Exception as e:
            print(f"Error uploading file {file_path}: {str(e)}")
            raise
    
    def process_transcripts(self, transcript_dir: str):
        """Process all transcript files and create vector store for Responses API."""
        try:
            # Create transcript directory if it doesn't exist
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Create vector store first
            self.create_vector_store("Transcript RAG System")
            
            # Check if directory has any transcript files
            transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith(('.json', '.txt'))]
            
            if not transcript_files:
                print(f"No transcript files found in {transcript_dir}")
                print("Please add transcript files (.json or .txt) to this directory first.")
                return
            
            # Upload all files to the vector store
            for filename in transcript_files:
                file_path = os.path.join(transcript_dir, filename)
                self.upload_file(file_path)
            
            print(f"Processed {len(self.uploaded_files)} files for Responses API")
        except Exception as e:
            print(f"Error in process_transcripts: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query using OpenAI's Responses API with file search."""
        if not self.vector_store:
            raise ValueError("RAG system not initialized. Please process transcripts first.")
        
        # Ensure question is properly encoded
        try:
            question_safe = question.encode('utf-8').decode('utf-8')
        except UnicodeError:
            question_safe = question.encode('ascii', errors='ignore').decode('ascii')
        
        try:
            # Create the input with conversation history
            input_content = question_safe
            if self.conversation_history:
                # Add recent conversation context
                recent_context = self.conversation_history[-4:]  # Last 2 exchanges
                context_str = ""
                for msg in recent_context:
                    # Ensure conversation history is properly encoded
                    try:
                        content_safe = str(msg['content']).encode('utf-8').decode('utf-8')
                        role_safe = str(msg['role']).encode('utf-8').decode('utf-8')
                        context_str += f"{role_safe}: {content_safe}\n"
                    except UnicodeError:
                        # Skip problematic messages
                        continue
                input_content = f"Previous conversation:\n{context_str}\n\nCurrent question: {question_safe}"
            
            # Use the Responses API with file search and vector store
            response = self.client.responses.create(
                model="gpt-4o",
                input=input_content,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [self.vector_store.id]
                    }
                ],
                include=["output[*].file_search_call.search_results"]  # Include search results
            )
            
            # Debug: Print response structure
            print(f"Response structure debug:")
            print(f"Response output length: {len(response.output) if response.output else 0}")
            for i, output_item in enumerate(response.output if response.output else []):
                print(f"Output item {i}: type={getattr(output_item, 'type', 'no_type')}")
                if hasattr(output_item, 'content'):
                    print(f"  Has content: {len(output_item.content) if output_item.content else 0} items")
                if hasattr(output_item, 'search_results'):
                    print(f"  Has search_results: {len(output_item.search_results) if output_item.search_results else 0} items")
                if hasattr(output_item, 'file_search_call') and hasattr(output_item.file_search_call, 'search_results'):
                    print(f"  file_search_call has search_results: {len(output_item.file_search_call.search_results) if output_item.file_search_call.search_results else 0} items")
                if hasattr(output_item, 'annotations'):
                    print(f"  Has annotations: {len(output_item.annotations) if output_item.annotations else 0} items")
            print("---")
            
            # Extract the answer
            if response.output and len(response.output) > 0:
                answer = ""
                sources = []
                
                # First, extract search results from file_search_call
                for output_item in response.output:
                    if hasattr(output_item, 'type') and output_item.type == 'file_search_call':
                        # Try different ways to access search results
                        search_results = None
                        if hasattr(output_item, 'search_results'):
                            search_results = output_item.search_results
                        elif hasattr(output_item, 'file_search_call') and hasattr(output_item.file_search_call, 'search_results'):
                            search_results = output_item.file_search_call.search_results
                        
                        if search_results:
                            print(f"Found {len(search_results)} search results")
                            for result in search_results:
                                filename = getattr(result, 'filename', 'Unknown')
                                file_id = getattr(result, 'file_id', 'unknown')
                                score = getattr(result, 'score', 0.0)
                                
                                # Extract content snippet
                                content_snippet = ""
                                if hasattr(result, 'content') and result.content:
                                    for content_item in result.content:
                                        if hasattr(content_item, 'text'):
                                            content_text = content_item.text
                                            content_snippet = content_text[:300] + "..." if len(content_text) > 300 else content_text
                                            break
                                
                                if content_snippet:
                                    sources.append({
                                        "file_id": file_id,
                                        "filename": filename,
                                        "citation": content_snippet,
                                        "type": "search_result",
                                        "score": score
                                    })
                                    print(f"Added source: {filename} with {len(content_snippet)} chars")
                
                # Then extract the answer from message
                for output_item in response.output:
                    if hasattr(output_item, 'type') and output_item.type == 'message':
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text'):
                                    # Extract the answer text
                                    try:
                                        answer = str(content_item.text).encode('utf-8').decode('utf-8')
                                    except UnicodeError:
                                        answer = str(content_item.text).encode('ascii', errors='ignore').decode('ascii')
                                    
                                    # Also check for annotations in the message content
                                    if hasattr(content_item, 'annotations') and content_item.annotations:
                                        for annotation in content_item.annotations:
                                            if hasattr(annotation, 'file_citation'):
                                                file_id = annotation.file_citation.file_id
                                                quote = getattr(annotation.file_citation, 'quote', '')
                                                
                                                # Find filename from uploaded files
                                                filename = "Unknown"
                                                for file_info in self.uploaded_files:
                                                    if file_info["file_id"] == file_id:
                                                        filename = file_info["filename"]
                                                        break
                                                
                                                try:
                                                    citation_text = str(quote).encode('utf-8').decode('utf-8')
                                                except UnicodeError:
                                                    citation_text = str(quote).encode('ascii', errors='ignore').decode('ascii')
                                                
                                                sources.append({
                                                    "file_id": file_id,
                                                    "filename": filename,
                                                    "citation": citation_text,
                                                    "type": "file_citation"
                                                })
                
                print(f"Final: answer length={len(answer)}, sources count={len(sources)}")
                
                if answer:
                    # Update conversation history with safe encoding
                    self.conversation_history.append({"role": "user", "content": question_safe})
                    self.conversation_history.append({"role": "assistant", "content": answer})
                    
                    return {
                        "answer": answer,
                        "sources": sources,
                        "response_id": response.id if hasattr(response, 'id') else None,
                        "mode": "Responses API with file_search"
                    }
                else:
                    return {
                        "answer": "No response generated from the API.",
                        "sources": [],
                        "response_id": None,
                        "mode": "Responses API (no content)"
                    }
                
        except Exception as e:
            print(f"Error with Responses API: {str(e)}")
            # Fallback to regular chat completion
            return self._fallback_query(question_safe)
    
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """Fallback method using regular chat completions."""
        try:
            # Create a simple prompt with file information
            files_context = f"I have access to {len(self.uploaded_files)} transcript files: "
            files_context += ", ".join([f["filename"] for f in self.uploaded_files])
            
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. {files_context}. Answer questions based on the transcript content you have access to. If you cannot find specific information, say so clearly."
                }
            ]
            
            # Add conversation history with safe encoding
            for msg in self.conversation_history[-4:]:  # Last 2 exchanges
                try:
                    content_safe = str(msg['content']).encode('utf-8').decode('utf-8')
                    role_safe = str(msg['role']).encode('utf-8').decode('utf-8')
                    messages.append({"role": role_safe, "content": content_safe})
                except UnicodeError:
                    # Skip problematic messages
                    continue
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Ensure response is properly encoded
            try:
                answer = response.choices[0].message.content.encode('utf-8').decode('utf-8')
            except UnicodeError:
                answer = response.choices[0].message.content.encode('ascii', errors='ignore').decode('ascii')
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return {
                "answer": answer,
                "sources": [{"filename": f["filename"], "citation": "Chat completion mode - general knowledge response"} for f in self.uploaded_files],
                "response_id": None,
                "mode": "Chat Completions (fallback)"
            }
            
        except Exception as fallback_error:
            error_msg = f"Error: Could not process query. {str(fallback_error)}"
            try:
                error_msg = error_msg.encode('utf-8').decode('utf-8')
            except UnicodeError:
                error_msg = error_msg.encode('ascii', errors='ignore').decode('ascii')
            
            return {
                "answer": error_msg,
                "sources": [],
                "response_id": None,
                "mode": "Error"
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def cleanup(self):
        """Clean up resources (optional)."""
        if self.vector_store:
            # Note: You might want to keep vector stores for reuse
            pass
    
    def list_vector_stores(self):
        """List all available vector stores."""
        try:
            vector_stores = self.client.vector_stores.list()
            return [
                {
                    "id": vs.id,
                    "name": vs.name,
                    "created_at": vs.created_at,
                    "file_count": vs.file_counts.completed if hasattr(vs, 'file_counts') else 0
                }
                for vs in vector_stores.data
            ]
        except Exception as e:
            print(f"Error listing vector stores: {str(e)}")
            return []
    
    def connect_to_vector_store(self, vector_store_id: str):
        """Connect to an existing vector store."""
        try:
            self.vector_store = self.client.vector_stores.retrieve(vector_store_id)
            
            # Get files in this vector store
            files = self.client.vector_stores.files.list(vector_store_id=vector_store_id)
            self.uploaded_files = []
            
            for file_obj in files.data:
                # Get file details
                file_details = self.client.files.retrieve(file_obj.id)
                self.uploaded_files.append({
                    "file_id": file_obj.id,
                    "filename": file_details.filename,
                    "file_path": f"vector_store/{file_details.filename}"
                })
            
            print(f"Connected to vector store: {self.vector_store.id}")
            print(f"Found {len(self.uploaded_files)} files in vector store")
            return True
        except Exception as e:
            print(f"Error connecting to vector store {vector_store_id}: {str(e)}")
            return False
    
    def new_conversation(self):
        """Start a new conversation (clear history)."""
        self.conversation_history = []
        print("Started new conversation") 