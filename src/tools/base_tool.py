from abc import ABC, abstractmethod
from typing import Dict, Any

class Tool(ABC):
    """Base class for executable tools that agents can call"""

    name: str = "base_tool"
    description: str = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool with given kwargs and return result"""
        raise NotImplementedError 