# flake8: noqa
from .specialized_agents import (
    EmailComposeAgent,
    ResearchPaperAgent,
    AcademicConceptsAgent,
    RedirectAgent,
    GeneralAgent
)
from .vision_agent import VisionAgent
from .planner_agent import PlannerAgent
from models.classification import PromptClassifier, AgentType

# Initialize the classifier
classifier = PromptClassifier()

# Initialize all agents
agents = {
    AgentType.EMAIL: EmailComposeAgent(),
    AgentType.RESEARCH: ResearchPaperAgent(),
    AgentType.ACADEMIC: AcademicConceptsAgent(),
    AgentType.REDIRECT: RedirectAgent(),
    AgentType.PLANNER: PlannerAgent(),
    AgentType.GENERAL: GeneralAgent(),
    AgentType.VISION: VisionAgent()
}

def determine_agent_type(message: str, has_attachment: bool=False) -> str:
    """
    Determine which agent should handle the message using the classification system.
    
    Args:
        message: The user's message to classify
        has_attachment: Whether the message contains an attachment
        
    Returns:
        The type of agent that should handle the message
    """
    if has_attachment:
        return AgentType.VISION
    result = classifier.classify_message(message)
    
    # Log classification details
    print(f"Classified as: {result.agent_type}")
    print(f"Confidence score: {result.confidence_score}")
    print(f"Matched keywords: {result.matched_keywords}")
    print(f"Alternative agents: {result.alternative_agents}")
    
    # Return the agent type
    return result.agent_type 