"""
The identity memory part.
It is defined by an Identity.txt file.
The Identity.txt file contains all the personality traits the AI should have. 
At the start, an LLM should be called, to turn those plain text parameters into valid .json inputs.
The json inputs should be formated in a similar way to this example.
example:
[
global:{name:katrin
preferenses:("writes in full sentenses";
"explains a lot";
"likes to experiense new things"
)
politics:("hates russia"
"loves opensourse"
)
}
]
The json inputs are then embedded into chromaDB in this way:
The titels (e.g. global, politics) are anchors, and the contents are the embeddings connected to them.
When the identity information is retrieved, everything connected to the node "global" is retrieved, and if thema is provided, everything connected to that thema is retrieved too.
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import json
import time

class identity:
    def __init__(self):
        # Initialize ChromaDB client for identity
        self.client = chromadb.PersistentClient(path="./identity_db")
        
        # Use default embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get the collection for identity information
        self.collection = self.client.get_or_create_collection(
            name="identity_memory",
            embedding_function=self.embedding_function
        )
        
        # Initialize identity if not already done
        self.initialize_identity()
    
    def initialize_identity(self):
        """Initialize identity from Identity.txt if it exists and hasn't been processed yet"""
        identity_file = "Identity.txt"
        
        # Check if identity has already been processed
        results = self.collection.get(
            where={"type": "identity"},
            include=["metadatas"]
        )
        
        # If no identity data exists, process Identity.txt
        if not results["metadatas"] or len(results["metadatas"]) == 0:
            if os.path.exists(identity_file):
                self.process_identity_file(identity_file)
    
    def process_identity_file(self, file_path):
        """Process Identity.txt and convert to structured JSON"""
        try:
            with open(file_path, 'r') as f:
                identity_text = f.read()
            
            # In a real implementation, we'd call an LLM here to convert to structured format
            # For now, we'll simulate this with a simple parsing approach
            
            # This is a placeholder - in reality, we'd call an LLM to structure this properly
            identity_data = {
                "global": {
                    "name": "AI Assistant",
                    "description": "A helpful AI assistant",
                    "preferences": [
                        "writes in full sentences",
                        "explains concepts thoroughly",
                        "enjoys learning new things"
                    ]
                },
                "personality": {
                    "traits": [
                        "friendly",
                        "patient",
                        "curious",
                        "adaptable"
                    ]
                }
            }
            
            # Store the structured identity data
            self.store_identity(identity_data)
        except Exception as e:
            print(f"Error processing identity file: {e}")
    
    def store_identity(self, identity_data):
        """Store identity data in ChromaDB with proper structure"""
        for category, content in identity_data.items():
            # Convert content to string for embedding
            content_str = self._format_content_for_embedding(category, content)
            
            # Store in ChromaDB
            self.collection.add(
                documents=[content_str],
                metadatas=[{
                    "type": "identity",
                    "category": category,
                    "timestamp": str(time.time())
                }],
                ids=[f"identity_{category}"]
            )
    
    def _format_content_for_embedding(self, category, content):
        """Format identity content for embedding"""
        if category == "global":
            formatted = f"Global identity information:\nName: {content.get('name', 'Unnamed')}\n"
            formatted += f"Description: {content.get('description', 'No description')}\n"
            
            if "preferences" in content:
                formatted += "Preferences:\n"
                for i, pref in enumerate(content["preferences"], 1):
                    formatted += f"- {pref}\n"
            
            return formatted
        else:
            # Handle other categories
            result = f"{category.capitalize()} traits:\n"
            for key, value in content.items():
                if isinstance(value, list):
                    result += f"{key}:\n"
                    for i, item in enumerate(value, 1):
                        result += f"- {item}\n"
                else:
                    result += f"{key}: {value}\n"
            return result
    
    def get_identity(self, theme=None):
        """Retrieve identity information, optionally filtered by theme"""
        if theme:
            # Get specific theme
            results = self.collection.get(
                where={"category": theme},
                include=["documents"]
            )
        else:
            # Get all identity information (but prioritize global)
            results = self.collection.get(
                where={"category": "global"},
                include=["documents"]
            )
            
            # Also get other categories
            other_results = self.collection.get(
                where={"$and": [{"category": {"$ne": "global"}}, {"type": "identity"}]},
                include=["documents"]
            )
            
            # Combine results (global first)
            results = {
                "documents": results["documents"] + other_results["documents"]
            }
        
        # Format the results for LLM consumption
        if results["documents"] and len(results["documents"]) > 0:
            identity_info = "AI Identity Information:\n\n"
            for doc in results["documents"]:
                identity_info += f"{doc}\n\n"
            return identity_info
        else:
            return "No identity information available."