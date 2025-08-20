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

The json inputs are then embedded into Weaviate in this way:
The titles (e.g. global, politics) are anchors, and the contents are the embeddings connected to them.
When the identity information is retrieved, everything connected to the node "global" is retrieved,
and if theme is provided, everything connected to that theme is retrieved too.

This module handles reading the Identity.txt, converting it via LLM, structuring the data,
and preparing it for embedding into Weaviate. The actual Weaviate interaction is handled
conceptually here, requiring weaviate-client and sentence-transformers libraries.
"""

from config import LLM_SOURSE, OPENROUTER_API_KEY, LOCAL_MODEL_PATH, GEMINI_API_KEY
import os
import json
import logging
import weaviate
from weaviate.embedded import EmbeddedOptions
import requests

# --- Configuration ---
IDENTITY_FILE_PATH = "Identity.txt"
WEAVIATE_CLASS_NAME = "Identity"
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


def embed_identity_to_weaviate(structured_identity: dict, client):
    """
    Embeds the structured identity data into Weaviate.
    Each top-level key is an anchor, and its value is embedded and stored.
    """
    logger.info("Starting embedding process for identity data...")
    
    # Create the class schema if it doesn't exist
    class_obj = {
        "class": WEAVIATE_CLASS_NAME,
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            {
                "name": "anchor",
                "dataType": ["string"],
            },
            {
                "name": "content",
                "dataType": ["text"],
            }
        ]
    }
    
    if not client.schema.exists(WEAVIATE_CLASS_NAME):
        client.schema.create_class(class_obj)
        logger.info(f"Weaviate class '{WEAVIATE_CLASS_NAME}' created.")

    # Process each anchor-content pair
    for anchor, content in structured_identity.items():
        # Create a textual representation of the content for embedding
        if isinstance(content, list):
            text_to_embed = "; ".join(content)
        elif isinstance(content, dict):
            text_to_embed = json.dumps(content)
        else:
            text_to_embed = str(content)

        # Generate embedding using sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedding = model.encode(text_to_embed).tolist()

        # Prepare data for Weaviate
        data_object = {
            "anchor": anchor,
            "content": text_to_embed
        }

        # Check if object already exists
        existing = client.query.get(WEAVIATE_CLASS_NAME, ["anchor"]) \
            .with_where({
                "path": ["anchor"],
                "operator": "Equal",
                "valueString": anchor
            }).do()
        
        if existing['data']['Get'][WEAVIATE_CLASS_NAME]:
            # Update existing object
            uuid = existing['data']['Get'][WEAVIATE_CLASS_NAME][0]['_additional']['id']
            client.data_object.update(
                data_object=data_object,
                class_name=WEAVIATE_CLASS_NAME,
                uuid=uuid,
                vector=embedding
            )
            logger.info(f"Updated anchor '{anchor}' in Weaviate.")
        else:
            # Create new object
            client.data_object.create(
                data_object=data_object,
                class_name=WEAVIATE_CLASS_NAME,
                vector=embedding
            )
            logger.info(f"Created anchor '{anchor}' in Weaviate.")

    logger.info("Identity data successfully embedded into Weaviate.")


# --- Main Identity Class ---

class IdentityMemory:
    """Manages the AI's identity, including loading, structuring, embedding, and retrieving."""

    def __init__(self):
        """Initializes the IdentityMemory by loading and embedding the identity."""
        logger.info("Initializing IdentityMemory...")
        self.client = None
        self._setup_weaviate()
        self._load_and_process_identity()

    def _setup_weaviate(self):
        """Sets up the Weaviate client."""
        try:
            # Using embedded Weaviate for local development
            self.client = weaviate.Client(
                embedded_options=EmbeddedOptions()
            )
            logger.info("Weaviate client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    def _load_and_process_identity(self):
        """Loads identity text, converts it, and embeds it."""
        try:
            raw_text = read_identity_file(IDENTITY_FILE_PATH)
            structured_data = convert_identity_to_json(raw_text)
            embed_identity_to_weaviate(structured_data, self.client)
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
        if not self.client:
            logger.warning("Identity memory (Weaviate client) not initialized.")
            return ""

        try:
            # Query Weaviate for the document with the matching anchor
            result = self.client.query.get(WEAVIATE_CLASS_NAME, ["content"]) \
                .with_where({
                    "path": ["anchor"],
                    "operator": "Equal",
                    "valueString": theme
                }) \
                .with_limit(1) \
                .do()

            if result['data']['Get'][WEAVIATE_CLASS_NAME]:
                retrieved_text = result['data']['Get'][WEAVIATE_CLASS_NAME][0]['content']
                logger.info(f"Retrieved identity information for theme '{theme}'.")
                return retrieved_text
            else:
                logger.info(f"No identity information found for theme '{theme}'.")
                return ""

        except Exception as e:
            logger.error(f"Error retrieving identity for theme '{theme}': {e}")
            return ""


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