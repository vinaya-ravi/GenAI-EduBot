from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import textwrap

logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    """Enum for different types of agents"""
    EMAIL = "email"
    RESEARCH = "research"
    ACADEMIC = "academic"
    REDIRECT = "redirect"
    GENERAL = "general"
    PLANNER = "planner"
    VISION = "vision"

class EmailType(str, Enum):
    """Enum for different types of emails"""
    EXTENSION_REQUEST = "extension_request"
    MEETING_REQUEST = "meeting_request"
    CUSTOM = "custom"
    GENERAL = "general"

class EmailField(BaseModel):
    """Model for an email input field"""
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    options: Optional[List[str]] = None

class EmailInputRequest(BaseModel):
    """Model for requesting input from the user for email generation"""
    email_type: EmailType
    description: str
    fields: List[EmailField]
    collected_fields: Dict[str, str] = Field(default_factory=dict)
    next_field: Optional[str] = None

class AgentProfile(BaseModel):
    """Profile defining an agent's characteristics and capabilities"""
    type: AgentType = Field(..., description="Type of agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of agent's capabilities")
    keywords: List[str] = Field(default=[], description="Keywords associated with this agent")
    prompt_template: str = Field(..., description="Template for agent's system prompt")
    capabilities: List[str] = Field(default=[], description="List of agent's capabilities")
    priority: int = Field(default=0, description="Priority level for classification (higher = more specific)")
    context_requirements: List[str] = Field(default=[], description="Required context elements for this agent")

class AlternativeAgent(BaseModel):
    """Model for alternative agent suggestions"""
    agent_type: AgentType
    confidence_score: float
    context_match_score: float = Field(default=0.0, description="Score indicating how well the context matches")

class ClassificationResult(BaseModel):
    """Model for classification results"""
    agent_type: AgentType
    confidence_score: float
    context_match_score: float = Field(default=0.0, description="Score indicating how well the context matches")
    matched_keywords: List[str] = Field(default_factory=list)
    alternative_agents: List[AlternativeAgent] = Field(default_factory=list)
    context_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of the conversation context")

class EmailTemplate:
    """Class for email templates with proper structure and formatting"""
    
    @staticmethod
    def format_email(template: str, context: Dict[str, Any] = None) -> str:
        """
        Format an email template with proper structure and boundaries
        
        Args:
            template: The email template to format
            context: Optional context to fill template placeholders
            
        Returns:
            Properly structured and formatted email
        """
        # Apply context values if provided
        if context:
            for key, value in context.items():
                placeholder = f"[{key}]"
                template = template.replace(placeholder, str(value))
        
        # Format with proper structure
        lines = template.strip().split('\n')
        formatted_lines = []
        
        # Check if the first line is a subject line
        if lines and "Subject:" in lines[0]:
            formatted_lines.append(lines[0])
            lines = lines[1:]
        
        in_list = False
        for line in lines:
            line = line.rstrip()
            
            # Check for list items and format them properly
            if line.strip().startswith('*'):
                if not in_list:
                    formatted_lines.append("")  # Add space before list
                    in_list = True
                # Indent list items and ensure proper spacing
                formatted_line = line.strip()
                formatted_lines.append(formatted_line)
            else:
                if in_list:
                    formatted_lines.append("")  # Add space after list
                    in_list = False
                
                # Format regular paragraphs
                if line.strip():
                    # Break long lines for better readability
                    if len(line) > 80:
                        wrapped = textwrap.fill(line, width=80)
                        formatted_lines.extend(wrapped.split('\n'))
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append("")  # Preserve empty lines
        
        # Join and clean up multiple blank lines
        formatted_email = '\n'.join(formatted_lines)
        formatted_email = re.sub(r'\n{3,}', '\n\n', formatted_email)
        
        return formatted_email
    
    @staticmethod
    def create_extension_request(
        student_name: str,
        professor_name: str,
        course_name: str,
        course_id: str,
        current_due_date: str,
        proposed_due_date: str,
        reason: str,
        student_id: str,
        student_email: str
    ) -> str:
        """
        Create a properly formatted email for requesting an extension
        """
        template = textwrap.dedent(f"""
        Subject: Extension Request - {student_name} - {course_name} - Term Paper

        Dear Professor {professor_name},

        I am writing to respectfully request an extension for the term paper in {course_name} (Course ID: {course_id}). The original due date is {current_due_date}.

        * I have recently been experiencing {reason}.
        * I would be grateful if I could have an extension until {proposed_due_date}. I am confident that with a few additional days, I can submit high-quality work.
        * I understand the importance of meeting deadlines and apologize for any inconvenience this may cause. I am committed to succeeding in your course.

        Thank you for your time and consideration.

        Sincerely,

        {student_name}
        {student_id}
        {student_email}
        """).strip()
        
        return EmailTemplate.format_email(template)
    
    @staticmethod
    def create_meeting_request(
        student_name: str,
        professor_name: str,
        course_name: str,
        purpose: str,
        proposed_times: List[str],
        student_id: str,
        student_email: str
    ) -> str:
        """
        Create a properly formatted email for requesting a meeting
        """
        times_formatted = '\n'.join([f"* {time}" for time in proposed_times])
        
        template = textwrap.dedent(f"""
        Subject: Meeting Request - {student_name} - {course_name}

        Dear Professor {professor_name},

        I hope this email finds you well. I am writing to request a meeting to discuss {purpose} for {course_name}.

        I would like to propose the following times for our meeting:

        {times_formatted}

        If none of these times work for you, please let me know when would be convenient for your schedule.

        Thank you for your time and consideration.

        Sincerely,

        {student_name}
        {student_id}
        {student_email}
        """).strip()
        
        return EmailTemplate.format_email(template)

class PromptClassifier:
    """Classifier for determining which agent should handle a prompt"""
    
    def __init__(self):
        # Initialize keyword dictionaries for each agent type
        self.keywords = {
            AgentType.EMAIL: [
                "email", "compose", "write", "draft", "send", "message",
                "professor", "instructor", "faculty", "reply", "respond",
                "extension", "request", "meeting", "appointment"
            ],
            AgentType.RESEARCH: [
                "research", "paper", "thesis", "dissertation", "study",
                "methodology", "analysis", "literature", "review", "citation",
                "reference", "bibliography", "data", "results", "findings"
            ],
            AgentType.ACADEMIC: [
                "explain", "concept", "theory", "definition", "understand",
                "learn", "topic", "subject", "course", "material", "example",
                "homework", "assignment", "problem", "solution"
            ],
            AgentType.REDIRECT: [
                "where", "find", "location", "website", "link", "resource",
                "information", "contact", "office", "department", "building",
                "service", "help", "support", "assistance"
            ],
            AgentType.GENERAL: [
                "unt", "university", "campus", "student", "program",
                "admission", "enrollment", "registration", "general",
                "information", "question", "help"
            ],
            AgentType.PLANNER: [
                "plan", "steps", "guide", "process", "end to end", "roadmap",
                "how do i", "what should i do", "strategy", "approach"
            ],
            AgentType.VISION: [
                "image", "picture", "photo", "what is this", "describe this", "see", "look at"
            ]
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000
        )
        
        # Create corpus for training vectorizer
        corpus = []
        for keywords in self.keywords.values():
            corpus.append(" ".join(keywords))
        
        # Fit vectorizer on keyword corpus
        self.vectorizer.fit(corpus)
        
        # Create keyword vectors
        self.keyword_vectors = {}
        for agent_type, keywords in self.keywords.items():
            keyword_text = " ".join(keywords)
            vector = self.vectorizer.transform([keyword_text])
            self.keyword_vectors[agent_type] = vector
    
        # Initialize conversation context
        self.conversation_context = {
            "current_agent": None,
            "previous_messages": [],
            "collected_context": {},
            "intent_history": []
        }
        
        # Initialize email templates
        self.email_templates = {
            EmailType.EXTENSION_REQUEST: EmailTemplate.create_extension_request,
            EmailType.MEETING_REQUEST: EmailTemplate.create_meeting_request
        }
        
        # Initialize email field definitions
        self.email_field_definitions = self._initialize_email_field_definitions()
        
        # Initialize current email input request
        self.current_email_request = None
    
    def _initialize_email_field_definitions(self) -> Dict[EmailType, List[EmailField]]:
        """Initialize the field definitions for each email type"""
        field_definitions = {}
        
        # Extension request fields
        field_definitions[EmailType.EXTENSION_REQUEST] = [
            EmailField(name="student_name", description="Your full name"),
            EmailField(name="professor_name", description="Professor's name"),
            EmailField(name="course_name", description="Course name"),
            EmailField(name="course_id", description="Course ID or number"),
            EmailField(name="current_due_date", description="Current due date of the assignment"),
            EmailField(name="proposed_due_date", description="Requested new due date"),
            EmailField(name="reason", description="Brief reason for requesting the extension"),
            EmailField(name="student_id", description="Your student ID number"),
            EmailField(name="student_email", description="Your university email address")
        ]
        
        # Meeting request fields
        field_definitions[EmailType.MEETING_REQUEST] = [
            EmailField(name="student_name", description="Your full name"),
            EmailField(name="professor_name", description="Professor's name"),
            EmailField(name="course_name", description="Course name"),
            EmailField(name="purpose", description="Purpose of the meeting"),
            EmailField(name="proposed_times", description="List of proposed meeting times (comma-separated)"),
            EmailField(name="student_id", description="Your student ID number"),
            EmailField(name="student_email", description="Your university email address")
        ]
        
        # Custom email fields
        field_definitions[EmailType.CUSTOM] = [
            EmailField(name="subject", description="Email subject"),
            EmailField(name="recipient", description="Recipient's name or title"),
            EmailField(name="body", description="Main content of the email"),
            EmailField(name="sender", description="Your name"),
            EmailField(name="sender_id", description="Your ID number", required=False),
            EmailField(name="sender_email", description="Your email address")
        ]
        
        # General email fields (simplified)
        field_definitions[EmailType.GENERAL] = [
            EmailField(name="subject", description="Email subject"),
            EmailField(name="recipient", description="Recipient's name or title"),
            EmailField(name="body", description="Main content of the email"),
            EmailField(name="sender", description="Your name"),
            EmailField(name="sender_email", description="Your email address")
        ]
        
        return field_definitions
    
    def start_email_input_collection(self, message: str) -> EmailInputRequest:
        """
        Start the email input collection process based on the user's message
        
        Args:
            message: The user's message
            
        Returns:
            EmailInputRequest with information about the next required input
        """
        # Determine the type of email needed
        email_type = self._determine_email_type(message)
        
        # Create a new input request
        self.current_email_request = EmailInputRequest(
            email_type=email_type,
            description=self._get_email_type_description(email_type),
            fields=self.email_field_definitions[email_type],
            next_field=self.email_field_definitions[email_type][0].name
        )
        
        # Pre-fill any information we can extract from the message
        self._prefill_fields_from_message(message)
        
        return self.current_email_request
    
    def _determine_email_type(self, message: str) -> EmailType:
        """Determine the type of email based on the user's message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["extension", "extend deadline", "more time", "delay"]):
            return EmailType.EXTENSION_REQUEST
        elif any(word in message_lower for word in ["meeting", "appointment", "discuss", "talk", "office hours"]):
            return EmailType.MEETING_REQUEST
        elif "custom" in message_lower or "specific" in message_lower:
            return EmailType.CUSTOM
        else:
            return EmailType.GENERAL
    
    def _get_email_type_description(self, email_type: EmailType) -> str:
        """Get a description for the email type"""
        descriptions = {
            EmailType.EXTENSION_REQUEST: "Email to request an extension for an assignment or paper",
            EmailType.MEETING_REQUEST: "Email to request a meeting with a professor",
            EmailType.CUSTOM: "Custom email with your own content",
            EmailType.GENERAL: "General purpose email"
        }
        return descriptions.get(email_type, "Email")
    
    def _prefill_fields_from_message(self, message: str) -> None:
        """Extract and prefill fields from the user's message"""
        if not self.current_email_request:
            return
            
        # Simple extraction logic - could be enhanced with NER or other NLP techniques
        message_lower = message.lower()
        
        # Common fields
        if "my name is" in message_lower:
            name_match = re.search(r"my name is ([^.,]+)", message_lower)
            if name_match:
                name = name_match.group(1).strip().title()
                self._add_field_value("student_name", name)
                self._add_field_value("sender", name)
        
        # Course information
        if "course" in message_lower:
            course_match = re.search(r"course (?:is|called|named) ([^.,]+)", message_lower)
            if course_match:
                self._add_field_value("course_name", course_match.group(1).strip().title())
        
        # Professor information
        if "professor" in message_lower:
            prof_match = re.search(r"professor ([^.,]+)", message_lower)
            if prof_match:
                self._add_field_value("professor_name", prof_match.group(1).strip().title())
                self._add_field_value("recipient", prof_match.group(1).strip().title())
        
        # Specific to extension requests
        if self.current_email_request.email_type == EmailType.EXTENSION_REQUEST:
            if "due" in message_lower:
                due_match = re.search(r"due (?:on|by|date is) ([^.,]+)", message_lower)
                if due_match:
                    self._add_field_value("current_due_date", due_match.group(1).strip())
            
            if "reason" in message_lower:
                reason_match = re.search(r"reason (?:is|being) ([^.]+)", message_lower)
                if reason_match:
                    self._add_field_value("reason", reason_match.group(1).strip())
    
    def _add_field_value(self, field_name: str, value: str) -> None:
        """Add a field value to the current email request"""
        if not self.current_email_request:
            return
            
        # Check if this field exists for the current email type
        if any(field.name == field_name for field in self.current_email_request.fields):
            self.current_email_request.collected_fields[field_name] = value
    
    def process_email_input(self, field_value: str) -> Union[EmailInputRequest, str]:
        """
        Process user input for the current email field
        
        Args:
            field_value: The value provided by the user for the current field
            
        Returns:
            Either an updated EmailInputRequest with the next field,
            or the generated email if all fields are collected
        """
        if not self.current_email_request:
            return "Please start the email process first."
        
        # Add the provided value to collected fields
        current_field = self.current_email_request.next_field
        
        # Special case for proposed_times which needs to be a list
        if current_field == "proposed_times":
            times_list = [time.strip() for time in field_value.split(',')]
            self.current_email_request.collected_fields[current_field] = times_list
        else:
            self.current_email_request.collected_fields[current_field] = field_value
        
        # Find the next required field that hasn't been filled
        next_field = None
        for field in self.current_email_request.fields:
            if field.required and field.name not in self.current_email_request.collected_fields:
                next_field = field.name
                break
        
        if next_field:
            # Update the next field and return the request
            self.current_email_request.next_field = next_field
            return self.current_email_request
        else:
            # All required fields collected, generate the email
            email = self.generate_email_from_request()
            self.current_email_request = None  # Reset for next time
            return email
    
    def generate_email_from_request(self) -> str:
        """Generate an email from the collected input request"""
        if not self.current_email_request:
            return "No email request in progress."
        
        email_type = self.current_email_request.email_type
        fields = self.current_email_request.collected_fields
        
        # Handle special case for proposed_times if it's a string
        if email_type == EmailType.MEETING_REQUEST and isinstance(fields.get("proposed_times"), str):
            fields["proposed_times"] = [time.strip() for time in fields["proposed_times"].split(',')]
        
        # Generate the email using the appropriate template
        if email_type == EmailType.EXTENSION_REQUEST:
            return EmailTemplate.create_extension_request(**fields)
        elif email_type == EmailType.MEETING_REQUEST:
            return EmailTemplate.create_meeting_request(**fields)
        else:
            # For custom or general emails, use the generic template
            template = textwrap.dedent(f"""
            Subject: {fields.get('subject', 'No Subject')}

            Dear {fields.get('recipient', 'Professor')},

            {fields.get('body', 'Email body goes here.')}

            Thank you for your time and consideration.

            Sincerely,

            {fields.get('sender', 'Student')}
            {fields.get('sender_id', '')}
            {fields.get('sender_email', '')}
            """).strip()
            
            return EmailTemplate.format_email(template)
    
    def generate_email(self, template_type: str, **kwargs) -> str:
        """
        Generate a properly formatted email using a template
        
        Args:
            template_type: Type of email template to use
            **kwargs: Context variables for the template
            
        Returns:
            Formatted email content
        """
        # Map string template type to EmailType enum
        email_type = None
        for et in EmailType:
            if et.value == template_type:
                email_type = et
                break
        
        if not email_type:
            # Default to general if template type not found
            email_type = EmailType.GENERAL
        
        # Check if all required fields are provided
        required_fields = [field.name for field in self.email_field_definitions.get(email_type, []) 
                          if field.required]
        
        missing_fields = [field for field in required_fields if field not in kwargs]
        
        if missing_fields:
            # If fields are missing, start the input collection process
            self.current_email_request = EmailInputRequest(
                email_type=email_type,
                description=self._get_email_type_description(email_type),
                fields=self.email_field_definitions[email_type],
                next_field=missing_fields[0]
            )
            
            # Add any provided fields
            for key, value in kwargs.items():
                if key in required_fields:
                    self.current_email_request.collected_fields[key] = value
            
            return f"Missing required fields for {email_type.value}: {', '.join(missing_fields)}. Please provide {missing_fields[0]}."
        
        # All required fields provided, generate the email
        if email_type == EmailType.EXTENSION_REQUEST:
            return self.email_templates[email_type](**kwargs)
        elif email_type == EmailType.MEETING_REQUEST:
            return self.email_templates[email_type](**kwargs)
        else:
            # For custom templates, format a given template string
            if "template" in kwargs:
                return EmailTemplate.format_email(kwargs.pop("template"), kwargs)
            
            # Default generic email if no specific template
            template = textwrap.dedent(f"""
            Subject: {kwargs.get('subject', 'No Subject')}

            Dear {kwargs.get('recipient', 'Professor')},

            {kwargs.get('body', 'Email body goes here.')}

            Thank you for your time and consideration.

            Sincerely,

            {kwargs.get('sender', 'Student')}
            {kwargs.get('sender_id', '')}
            {kwargs.get('sender_email', '')}
            """).strip()
            
            return EmailTemplate.format_email(template, kwargs)
    
    def handle_email_request(self, message: str) -> Tuple[str, Optional[EmailInputRequest]]:
        """
        Handle a request to compose an email
        
        Args:
            message: The user's message
            
        Returns:
            A tuple containing a response message and an optional EmailInputRequest
        """
        # Check if we're already in the process of collecting email inputs
        if self.current_email_request and self.current_email_request.next_field:
            # User is providing a value for the current field
            result = self.process_email_input(message)
            
            if isinstance(result, EmailInputRequest):
                # Still collecting inputs
                next_field = next((f for f in self.current_email_request.fields 
                                if f.name == result.next_field), None)
                
                return (f"Please provide: {next_field.description}", result)
            else:
                # Email is generated
                return (f"Here's your email:\n\n{result}", None)
        else:
            # Start a new email collection process
            email_request = self.start_email_input_collection(message)
            next_field = next((f for f in email_request.fields 
                            if f.name == email_request.next_field), None)
            
            # Check if we have prefilled fields
            if email_request.collected_fields:
                prefilled = ", ".join([f"{k}: {v}" for k, v in email_request.collected_fields.items()])
                return (f"I'll help you write a {email_request.description}. I've identified the following information: {prefilled}\n\nPlease provide: {next_field.description}", email_request)
            else:
                return (f"I'll help you write a {email_request.description}. Please provide: {next_field.description}", email_request)

    def _analyze_context(self, message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze the conversation context to understand the current state"""
        context_analysis = {
            "current_topic": None,
            "intent": None,
            "required_context": set(),
            "missing_context": set(),
            "context_confidence": 0.0
        }
        
        # Analyze conversation history for context
        if conversation_history:
            # Get the last few messages for context
            recent_messages = conversation_history[-3:]  # Look at last 3 messages
            
            # Combine recent messages for context analysis
            context_text = " ".join([msg["content"] for msg in recent_messages])
            
            # Basic intent detection
            if any(word in message.lower() for word in ["email", "write", "send"]):
                context_analysis["intent"] = "email_composition"
                context_analysis["required_context"].update(["recipient", "purpose"])
            elif any(word in message.lower() for word in ["explain", "understand", "concept"]):
                context_analysis["intent"] = "concept_explanation"
                context_analysis["required_context"].update(["subject", "concept"])
            elif any(word in message.lower() for word in ["find", "where", "location"]):
                context_analysis["intent"] = "resource_location"
                context_analysis["required_context"].update(["resource_type", "specific_need"])
            
            # Check for missing context
            for required in context_analysis["required_context"]:
                if required not in self.conversation_context["collected_context"]:
                    context_analysis["missing_context"].add(required)
            
            # Calculate context confidence
            context_analysis["context_confidence"] = 1.0 - (len(context_analysis["missing_context"]) / 
                                                          max(1, len(context_analysis["required_context"])))
        
        return context_analysis
    
    def _calculate_context_match_score(self, agent_type: AgentType, context_analysis: Dict[str, Any]) -> float:
        """Calculate how well an agent matches the current context"""
        base_score = 0.0
        
        # Check if agent type matches the intent
        if context_analysis["intent"]:
            if agent_type == AgentType.EMAIL and context_analysis["intent"] == "email_composition":
                base_score += 0.4
            elif agent_type == AgentType.ACADEMIC and context_analysis["intent"] == "concept_explanation":
                base_score += 0.4
            elif agent_type == AgentType.REDIRECT and context_analysis["intent"] == "resource_location":
                base_score += 0.4
        
        # Add context confidence score
        base_score += context_analysis["context_confidence"] * 0.3
        
        # Add priority bonus for specific agents
        if agent_type in [AgentType.EMAIL, AgentType.ACADEMIC, AgentType.RESEARCH]:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def classify_message(self, message: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> ClassificationResult:
        """
        Classify a message to determine which agent should handle it.
        
        Args:
            message: The user's message to classify
            conversation_history: Optional list of previous messages for context
            
        Returns:
            ClassificationResult containing the best matching agent and alternatives
        """
        # Update conversation history
        if conversation_history:
            self.conversation_context["previous_messages"] = conversation_history
        
        # Analyze context
        context_analysis = self._analyze_context(message, conversation_history or [])
        
        # Transform message
        message_vector = self.vectorizer.transform([message])
        
        # Calculate similarities with each agent's keywords
        similarities = {}
        context_scores = {}
        for agent_type, keyword_vector in self.keyword_vectors.items():
            # Calculate keyword similarity
            similarity = cosine_similarity(message_vector, keyword_vector)[0][0]
            
            # Calculate context match score
            context_score = self._calculate_context_match_score(agent_type, context_analysis)
            
            # Combine scores (70% keyword similarity, 30% context match)
            combined_score = (similarity * 0.7) + (context_score * 0.3)
            
            similarities[agent_type] = combined_score
            context_scores[agent_type] = context_score
        
        # Sort by combined score
        sorted_agents = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get best match and alternatives
        best_match = sorted_agents[0]
        alternatives = sorted_agents[1:3]  # Get next 2 best matches
        
        # Find matched keywords
        message_words = set(message.lower().split())
        matched_keywords = []
        for word in message_words:
            for keywords in self.keywords.values():
                if word in keywords:
                    matched_keywords.append(word)
        
        # Create alternative agent list with context scores
        alternative_agents = [
            AlternativeAgent(
                agent_type=agent_type,
                confidence_score=float(score),
                context_match_score=float(context_scores[agent_type])
            )
            for agent_type, score in alternatives
        ]
        
        # Update conversation context
        self.conversation_context["current_agent"] = best_match[0]
        self.conversation_context["intent_history"].append(context_analysis["intent"])
        
        return ClassificationResult(
            agent_type=best_match[0],
            confidence_score=float(best_match[1]),
            context_match_score=float(context_scores[best_match[0]]),
            matched_keywords=matched_keywords,
            alternative_agents=alternative_agents,
            context_analysis=context_analysis
        ) 