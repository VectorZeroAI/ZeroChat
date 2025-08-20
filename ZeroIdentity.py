# ZeroIdentity.py

"""
The identity memory part.
It is defined by an Identity.txt file.
The Identity.txt file contains all the personality traits the AI should have.
At the start, an LLM should be called, to turn those plain text parameters into valid .json inputs.
The json inputs should be formatted in a similar way to this example.
example:
{
  "global": {
    "name": "katrin",
    "preferences": [
      "writes in full sentences",
      "explains a lot",
      "likes to experience new things"
    ],
    "politics": [
      "hates russia",
      "loves opensource"
    ]
  }
}

The json inputs are then embedded into chromaDB in this way:
The titles (e.g. global, politics) are anchors, and the contents are the embeddings connected to them.
When the identity information is retrieved, everything connected to the node "global" is retrieved,
and if theme is provided, everything connected to that theme is retrieved too.

This module handles reading the Identity.txt, converting it via LLM, structuring the data,
and preparing it for embedding into ChromaDB. The actual ChromaDB interaction is handled
conceptually here, requiring chromadb and sentence-transformers libraries.
"""

from config import LLM_SOURSE, OPENROUTER_API_KEY, LOCAL_MODEL_PATH, GEMINI_API_KEY
import os
import json
import logging
# Assuming call_LLM is refactored to be importable from ZeroMain
# This might require moving it to a shared utils file or adjusting ZeroMain.py structure
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
IDENTITY_FILE_PATH = "Identity.txt"
CHROMA_DB_PATH = "./chroma_db_identity" # Local directory for ChromaDB persistence
CHROMA_COLLECTION_NAME = "identity_embeddings"
# Using a Sentence Transformer model for embeddings
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# ---------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def call_LLM(prompt: str) -> str:
    if LLM_SOURSE == "OPENROUTER":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    elif LLM_SOURSE == "LOCAL":
        # HuggingFace transformers local pipeline
        from transformers import pipeline
        local_llm = pipeline("text-generation", model=LOCAL_MODEL_PATH)
        result = local_llm(prompt, max_length=512, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]

    elif LLM_SOURSE == "GEMINI":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text

    else:
        raise ValueError("Invalid config: LLM_SOURSE not recognized.")



def read_identity_file(filepath: str) -> str:
    """Reads the raw identity text from Identity.txt."""
    if not os.path.exists(filepath):
        logger.error(f"Identity file not found at {filepath}")
        raise FileNotFoundError(f"Identity file not found at {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    logger.info("Identity file read successfully.")
    return content

def convert_identity_to_json(raw_identity_text: str) -> dict:
    """
    Uses the LLM to convert raw identity text into structured JSON.
    Returns the parsed JSON dictionary.
    """
    # Example prompt structure based on the description
    prompt = f"""
    Please convert the following list of personality traits and facts into a structured JSON format.
    The top-level keys should represent categories (like 'global', 'preferences', 'politics').
    The values for these keys should be objects or arrays containing the relevant details.
    The 'global' section should contain core identifiers like 'name'.
    Other sections can be named based on the content (e.g., 'preferences', 'politics').

    Example Input:
    name: katrin
    writes in full sentences; explains a lot; likes to experience new things
    hates russia; loves opensource

    Example Output:
    {{
      "global": {{
        "name": "katrin"
      }},
      "preferences": [
        "writes in full sentences",
        "explains a lot",
        "likes to experience new things"
      ],
      "politics": [
        "hates russia",
        "loves opensource"
      ]
    }}

    Now, convert this input:
    {raw_identity_text}

    Please provide only the JSON output, nothing else.
    """

    logger.info("Calling LLM to convert identity text to JSON...")
    llm_response = call_LLM(prompt)
    logger.info("LLM response received.")

    # Attempt to parse the LLM's response as JSON
    try:
        # LLMs sometimes add markdown code block ticks
        if llm_response.startswith("```json"):
            llm_response = llm_response[7:]
        if llm_response.endswith("```"):
            llm_response = llm_response[:-3]

        structured_data = json.loads(llm_response)
        logger.info("Identity text successfully converted to JSON structure.")
        return structured_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"LLM Response was: {llm_response}")
        # Return a default or raise an error depending on desired robustness
        raise ValueError("LLM did not return valid JSON for identity.") from e


def embed_identity_to_chromadb(structured_identity: dict, chroma_client, collection):
    """
    Embeds the structured identity data into ChromaDB.
    Each top-level key is an anchor, and its value is embedded and stored.
    """
    logger.info("Starting embedding process for identity data...")
    # Initialize sentence transformer model for embeddings
    # Note: The embedding function is set on the collection, so we don't need to instantiate it here
    # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    documents = []
    metadatas = []
    ids = []

    for anchor, content in structured_identity.items():
        # Create a textual representation of the content for embedding
        # If it's a list, join items. If it's a dict, stringify it or process further.
        if isinstance(content, list):
            # Join list items into a single string for embedding
            text_to_embed = "; ".join(content)
        elif isinstance(content, dict):
            # For nested dicts, could serialize or extract specific parts.
            # Simple string representation for now.
            text_to_embed = json.dumps(content)
        else:
            # For strings or other primitives
            text_to_embed = str(content)

        # Prepare data for ChromaDB
        doc_id = f"identity_{anchor}" # Unique ID for this anchor
        documents.append(text_to_embed)
        metadatas.append({"anchor": anchor}) # Store the anchor as metadata
        ids.append(doc_id)

        logger.info(f"Prepared anchor '{anchor}' for embedding.")

    if documents:
        # Add documents to the ChromaDB collection
        # The collection's embedding function will automatically embed the 'documents'
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Identity data successfully embedded into ChromaDB.")
    else:
        logger.warning("No identity data found to embed.")


# --- Main Identity Class ---

class IdentityMemory:
    """Manages the AI's identity, including loading, structuring, embedding, and retrieving."""

    def __init__(self):
        """Initializes the IdentityMemory by loading and embedding the identity."""
        logger.info("Initializing IdentityMemory...")
        self.client = None
        self.collection = None
        self._setup_chromadb()
        self._load_and_process_identity()

    def _setup_chromadb(self):
        """Sets up the ChromaDB client and collection."""
        try:
            # Persistent client
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            # Get or create the collection
            # Define the embedding function for this collection
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_func
            )
            logger.info("ChromaDB client and collection initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _load_and_process_identity(self):
        """Loads identity text, converts it, and embeds it."""
        try:
            raw_text = read_identity_file(IDENTITY_FILE_PATH)
            structured_data = convert_identity_to_json(raw_text)
            embed_identity_to_chromadb(structured_data, self.client, self.collection)
            logger.info("Identity processing and embedding complete.")
        except FileNotFoundError:
            logger.warning("Identity file not found. Identity memory will be empty.")
        except Exception as e:
             # Catch errors during LLM call or JSON parsing
             logger.error(f"Error processing identity: {e}. Identity memory might be incomplete.")
             # Depending on requirements, you might want to stop or continue


    def retrieve(self, theme: str = "global") -> str:
        """
        Retrieves identity information related to a given theme/anchor.
        Defaults to retrieving 'global' identity information.
        Returns a string representation of the retrieved data.
        """
        if not self.collection:
            logger.warning("Identity memory (ChromaDB collection) not initialized.")
            return ""

        try:
            # Query ChromaDB for the document with the matching anchor metadata
            # Using `get` with `where` is appropriate for exact metadata matches
            results = self.collection.get(
                # Filter by metadata to get the specific anchor
                where={"anchor": theme},
                # We expect only one document per anchor
                limit=1
            )

            if results['ids'] and len(results['ids']) > 0:
                # Assuming one result per anchor, get the first document
                # The 'documents' list contains the original text that was embedded
                retrieved_text = results['documents'][0]
                logger.info(f"Retrieved identity information for theme '{theme}'.")
                return retrieved_text
            else:
                logger.info(f"No identity information found for theme '{theme}'.")
                return "" # Return empty string if not found, as per original description logic

        except Exception as e:
            logger.error(f"Error retrieving identity for theme '{theme}': {e}")
            return "" # Return empty string on error


# --- Initialize Identity at Module Load ---
# This creates the instance when the module is imported.
indentity = IdentityMemory()
logger.info("IdentityMemory instance 'indentity' created.")

# Example usage within this module (optional, for testing)
if __name__ == "__main__":
    # This part runs if the script is executed directly
    # It will trigger the __init__ which loads and processes the identity
    print("ZeroIdentity module executed directly. Initializing identity...")
    # Access the initialized instance
    identity_instance = indentity
    # Example: Retrieve global identity
    global_info = identity_instance.retrieve("global")
    print(f"Retrieved Global Identity Info:\n{global_info}\n---")
    # Example: Retrieve politics identity
    politics_info = identity_instance.retrieve("politics")
    print(f"Retrieved Politics Identity Info:\n{politics_info}\n---")
    # Example: Retrieve non-existent theme
    unknown_info = identity_instance.retrieve("hobbies")
    print(f"Retrieved Unknown Theme Info:\n'{unknown_info}'\n---")