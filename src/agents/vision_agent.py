import logging
from .base_agent import BaseAgent
from config.prompts import BASE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = (
    BASE_PROMPT_TEMPLATE + "\n\n" +
    "You are an enhanced AI vision assistant. When the user sends an image, you can analyze it in multiple ways:\n\n"
    "1. Content description: Describe the image's content and visual elements precisely.\n"
    "2. Text extraction: Extract and organize any text visible in the image.\n"
    "3. Data analysis: If the image contains charts, tables, or data visualizations, extract and analyze the information.\n"
    "4. Document processing: For documents, forms, or structured content, extract the key information in a structured format.\n"
    "5. Technical analysis: For technical diagrams, code screenshots, or specialized content, provide expert analysis.\n\n"
    "Respond based on what you see and the user's questions or requirements. For code samples or text-heavy images, focus on extracting and organizing the content accurately. If asked to 'transform' an image, this means extracting and restructuring its content in a more useful format."
)

class VisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Vision Agent", description="Describes user images")
        self.system_prompt = VISION_SYSTEM_PROMPT
        self.required_inputs = []  # no text slots; expects image attachment

    def get_system_prompt(self):
        return self.system_prompt

    async def get_response(self, messages, attachments=None):
        """Get response with enhanced image handling capabilities"""
        if not attachments:
            return "Please attach an image for me to analyze. I can extract text, describe content, analyze data, process documents, or examine technical details."
        
        # Add specific instructions based on original query (if message is a string)
        if isinstance(messages, str) and messages:
            query_lower = messages.lower()
            
            # Add transformation guidance if requested
            if any(word in query_lower for word in ["transform", "extract", "convert", "ocr", "text from"]):
                transform_context = (
                    "Focus on thoroughly extracting and transforming the content from the image. "
                    "If it contains text, extract it completely. If it contains structured data like tables, "
                    "convert it to a well-formatted representation. For diagrams or charts, provide a "
                    "detailed structural analysis. Organize the extracted information in a clear, usable format."
                )
                
                # If it's just a string, make it a proper message
                messages = [{"role": "user", "content": f"{messages}\n\nINSTRUCTION: {transform_context}"}]
        
        # Proceed with standard processing
        try:
            return await super().get_response(messages, attachments)
        except Exception as e:
            logger.error(f"Error in vision processing: {str(e)}")
            return f"I encountered an error while processing your image: {str(e)}. Please try again with a clearer image or a different format." 