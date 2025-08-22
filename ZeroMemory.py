#ZeroMemory.py
"""
This File is the high orchestrator of all the memory parts, bringing them all together into the "full" funktion.
It now handles both identity information and conversation context.
"""

from ZeroIdentity import identity as IdentitySystem
import chromadb
from chromadb.utils import embedding_functions
import uuid
import time

class memory:
    def __init__(self):
        # Initialize identity system
        self.identity = IdentitySystem()
        
        # Initialize ChromaDB client for conversation history
        self.client = chromadb.PersistentClient(path="./memory_db")
        
        # Use default embedding function (all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get the collection for conversation history
        self.collection = self.client.get_or_create_collection(
            name="conversation_memory",
            embedding_function=self.embedding_function
        )
    
    def add_user_prompt(self, prompt):
        """Add a user prompt to the database with metadata"""
        # Generate a unique ID for this interaction
        interaction_id = str(uuid.uuid4())
        
        # Store the timestamp
        timestamp = time.time()
        
        # Add to ChromaDB
        self.collection.add(
            documents=[prompt],
            metadatas=[{
                "type": "user",
                "timestamp": str(timestamp),
                "interaction_id": interaction_id
            }],
            ids=[f"user_{interaction_id}"]
        )
        
        return interaction_id
    
    def add_ai_response(self, response, interaction_id):
        """Add an AI response to the database, linked to the user prompt"""
        # Store the timestamp
        timestamp = time.time()
        
        # Add to ChromaDB
        self.collection.add(
            documents=[response],
            metadatas=[{
                "type": "ai",
                "timestamp": str(timestamp),
                "interaction_id": interaction_id
            }],
            ids=[f"ai_{interaction_id}"]
        )
    
    def get_similar_conversations(self, query, n_results=2):
        """
        Retrieve the n_results most similar user prompts and their corresponding AI responses
        """
        try:
            # First, find similar user prompts
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"type": "user"},
                include=["documents", "metadatas", "ids"]
            )
            
            # Check if we got any results
            if not results["metadatas"] or not results["metadatas"][0]:
                return []
            
            similar_conversations = []
            for i in range(len(results["ids"][0])):
                user_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                
                # Extract the actual interaction ID (remove "user_" prefix)
                interaction_id = metadata.get("interaction_id", user_id[5:])
                user_prompt = results["documents"][0][i]
                timestamp = metadata.get("timestamp", "unknown")
                
                # Get the corresponding AI response
                ai_results = self.collection.get(
                    where={"interaction_id": interaction_id, "type": "ai"},
                    include=["documents"]
                )
                
                # Handle case where AI response might not be found
                ai_response = ai_results["documents"][0] if ai_results["documents"] and len(ai_results["documents"]) > 0 else "No response found"
                
                similar_conversations.append({
                    "user": user_prompt,
                    "ai": ai_response,
                    "interaction_id": interaction_id,
                    "timestamp": timestamp
                })
            
            return similar_conversations
        except Exception as e:
            print(f"Error retrieving similar conversations: {e}")
            return []
    
    def format_timestamp(self, timestamp_str):
        """Format timestamp for better readability"""
        try:
            timestamp = float(timestamp_str)
            # Convert to a more readable format
            seconds_ago = time.time() - timestamp
            minutes_ago = seconds_ago / 60
            hours_ago = minutes_ago / 60
            days_ago = hours_ago / 24
            
            if days_ago > 1:
                return f"{int(days_ago)} days ago"
            elif hours_ago > 1:
                return f"{int(hours_ago)} hours ago"
            elif minutes_ago > 1:
                return f"{int(minutes_ago)} minutes ago"
            else:
                return "just now"
        except:
            return "unknown time"
    
    def full(self, query):
        """
        Retrieve full context for the current query - combines identity information
        and relevant conversation history
        """
        # Get identity information (global + other themes)
        identity_info = self.identity.get_identity()
        
        # Get relevant conversation history
        similar_conversations = self.get_similar_conversations(query)
        
        # Format the complete context
        context = f"{identity_info}\n\n"
        context += "Relevant past conversations (most relevant first):\n"
        
        if not similar_conversations:
            context += "No relevant past conversations found.\n"
        else:
            for i, conv in enumerate(similar_conversations, 1):
                context += f"\nConversation #{i} (from {self.format_timestamp(conv['timestamp'])}):\n"
                context += f"User: {conv['user']}\n"
                context += f"AI: {conv['ai']}\n"
        
        return context