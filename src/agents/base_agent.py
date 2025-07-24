from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import base64
import time
from openai import OpenAI
import os

from config.settings import (
    MODEL_ID,
    INFERENCE_SERVER_URL,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    MAX_TOKENS,
    TEMPERATURE
)
from utils.vector_db import VectorDBManager

logger = logging.getLogger(__name__)

# Initialize OpenAI client with vLLM API endpoint
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require an actual API key
    base_url=INFERENCE_SERVER_URL,
    timeout=REQUEST_TIMEOUT
)

# Initialize vector database manager
vector_db = VectorDBManager()

# Helper function to encode image to base64
def encode_image_to_base64(image_data):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_data).decode('utf-8')

class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self.required_inputs = []
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None
        # If the subclass sets query_type and didn't explicitly set required_inputs,
        # auto-generate required_inputs from the Pydantic model fields
        if hasattr(self, "query_type") and not self.required_inputs and getattr(self, "query_type") is not None:
            try:
                model_cls = getattr(self, "query_type")
                # Handle Pydantic v1 (__fields__) and v2 (model_fields)
                if hasattr(model_cls, "__fields__"):
                    fields_map = model_cls.__fields__  # type: ignore
                else:
                    fields_map = model_cls.model_fields  # type: ignore
                for fname, f in fields_map.items():
                    # Determine if the field is required
                    required = getattr(f, "required", False)
                    if required:
                        question = f.description or f"Please provide {fname.replace('_',' ')}."
                        self.required_inputs.append({"key": fname, "question": question})
            except Exception as e:
                logging.getLogger(__name__).warning(f"Auto-generation of required_inputs failed: {e}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent"""
        pass
    
    def get_relevant_context(self, query: str) -> str:
        """Get relevant context from vector database"""
        try:
            relevant_docs = vector_db.get_relevant_documents(query)
            if relevant_docs:
                return "\n\nRelevant context:\n" + "\n".join(relevant_docs)
            return ""
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""
    
    def needs_additional_input(self) -> bool:
        """Check if the agent needs more information from the user"""
        if not self.required_inputs:
            return False
            
        for input_item in self.required_inputs:
            if input_item["key"] not in self.collected_inputs:
                return True
        return False
    
    def process_input(self, message: str) -> Dict[str, Any]:
        """Process the user input and decide what to do next"""
        result = {"type": "response", "content": None, "next_question": None}
        
        # If waiting for a specific input, collect it
        if self.waiting_for_input and self.current_input_key:
            self.collected_inputs[self.current_input_key] = message
            self.waiting_for_input = False
            self.current_input_key = None
        
        # Check if we need more inputs
        if self.needs_additional_input():
            # Find the next required input
            for input_item in self.required_inputs:
                if input_item["key"] not in self.collected_inputs:
                    self.waiting_for_input = True
                    self.current_input_key = input_item["key"]
                    result["type"] = "input_request"
                    result["next_question"] = input_item["question"]
                    break
        else:
            # All inputs collected, ready for final response
            result["type"] = "final_response"
            
        return result
    
    async def get_response(self, messages: List[Dict[str, str]], attachments=None) -> str:
        """Get a response from the LLM using this agent's specialized prompt"""
        try:
            # Process attachments if any (for multimodal input)
            if attachments and len(attachments) > 0:
                try:
                    # Create a list to hold content items (for multimodal input)
                    user_content = []
                    
                    # Add text message if present in the last message
                    if messages[-1]["content"]:
                        user_content.append({"type": "text", "text": messages[-1]["content"]})
                    
                    # Process each attachment
                    for attachment in attachments:
                        # Get the image bytes
                        image_data = await attachment.get_bytes()
                        logger.info(f"Received image attachment of size: {len(image_data)} bytes")
                        
                        # Encode the image to base64
                        base64_image = encode_image_to_base64(image_data)
                        
                        # Add the image to content
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    
                    # Update the last message with multimodal content
                    messages[-1]["content"] = user_content
                
                except Exception as img_err:
                    logger.exception("Error processing image attachment:")
                    if not messages[-1]["content"]:
                        messages[-1]["content"] = f"[Image attachment error: {str(img_err)}]"
            
            # Ensure messages is a list of dictionaries with role and content
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(messages)}]
            
            # Add system prompt if not present
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": self.get_system_prompt()})
            
            # Get relevant context from vector database for the last user message
            if messages[-1]["role"] == "user":
                context = self.get_relevant_context(messages[-1]["content"])
                if context:
                    messages[-1]["content"] += context
            
            # First, test if the server is reachable with a quick timeout
            try:
                test_client = OpenAI(
                    api_key="EMPTY",
                    base_url=INFERENCE_SERVER_URL,
                    timeout=3.0  # Very short timeout just for connectivity check
                )
                test_client.models.list()
                logger.info("LLM server is reachable")
            except Exception as conn_err:
                logger.error(f"Cannot reach LLM server: {str(conn_err)}")
                return "I'm having trouble connecting to the AI service. The LLM server appears to be unreachable. Please check that it's running at the configured URL."
                
            # If server is reachable, proceed with retries
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    # Make the inference request using the OpenAI client
                    response = client.chat.completions.create(
                        model=MODEL_ID,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE
                    )
                    
                    # Extract and log the response content
                    reply = response.choices[0].message.content
                    logger.info(f"Inference response from {self.name} agent")
                    return reply
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        # If we've exhausted retries, return a user-friendly error
                        error_message = (
                            "I apologize, but I'm having trouble generating a response. "
                            "This could be due to:\n"
                            "1. The request is taking too long\n"
                            "2. The server is overloaded\n"
                            "3. The query is too complex\n\n"
                            "Please try:\n"
                            "1. Simplifying your question\n"
                            "2. Breaking it into smaller parts\n"
                            "3. Trying again in a few moments\n\n"
                            f"Error details: {str(last_error)}"
                        )
                        return error_message
                        
        except Exception as e:
            error_message = (
                "I apologize, but I'm having trouble processing your request. "
                "This could be due to:\n"
                "1. The AI service is not running\n"
                "2. Network connectivity issues\n"
                "3. Service configuration problems\n\n"
                f"Error details: {str(e)}\n\n"
                "Please try again in a few moments or contact support if the issue persists."
            )
            return error_message

    def reset(self):
        """Reset the agent state for a new conversation"""
        self.collected_inputs = {}
        self.waiting_for_input = False
        self.current_input_key = None 

    def is_waiting_for_input(self) -> bool:
        return self.current_input_key is not None 