from typing import Dict, Any, Optional
import logging
from .base_agent import BaseAgent
from models.query_models import QueryResponse
from models.classification import PromptClassifier, AgentType

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """Super agent that decomposes a high-level user goal into sub-tasks and dispatches to specialized agents."""

    def __init__(self):
        super().__init__(name="Planner", description="High-level planner agent")
        self.system_prompt = (
            "You are an advanced AI planning assistant. Given a user's overall goal, you break it into logical steps, "
            "decide which domain expert (Email, Research, Academic Concept, Redirect) should handle each step, "
            "gather their outputs, then compile a clear final answer."
        )
        # Ask for goal first
        self.required_inputs = [
            {"key": "goal", "question": "What is your overall goal?"}
        ]
        # Internal classifier to route tasks
        self.classifier = PromptClassifier()

    def get_system_prompt(self) -> str:
        return self.system_prompt

    async def get_response(self, messages, attachments=None):
        """Once we have the goal, plan and call sub-agents."""
        if not self.collected_inputs.get("goal"):
            return "Please provide your goal first."

        goal = self.collected_inputs["goal"]
        # Use classifier to decide which agent is best
        from agents.registry import agents  # late import to avoid circular dep
        classification = self.classifier.classify_message(goal)
        chosen_agent_type: AgentType = classification.agent_type
        if chosen_agent_type == AgentType.PLANNER:
            # Fallback to general if planner loops
            chosen_agent_type = AgentType.GENERAL
        sub_agent = agents[chosen_agent_type]
        sub_agent.reset()
        # Feed the goal to sub-agent
        result = sub_agent.process_input(goal)
        if result["type"] == "input_request":
            # propagate question to user
            question = result["next_question"]
            # store sub-agent reference in collected_inputs for follow-up (simple)
            self.current_sub_agent = sub_agent
            return question
        else:
            response = await sub_agent.get_response(goal)
            summary = (
                f"### Plan Execution Summary\n"
                f"Goal: {goal}\n\n"
                f"Delegated to **{chosen_agent_type.value}** agent.\n\n"
                f"---\n{response}"
            )
            return summary 