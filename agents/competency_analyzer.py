"""Competency analyzer agent for analyzing promotion requirements."""
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from agents.base import BaseAgent
from orchestrator.state import State


class CompetencyAnalyzerAgent(BaseAgent):
    """Agent that analyzes competency requirements for target promotion level."""
    
    def __init__(self):
        super().__init__(
            name="competency_analyzer"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a Senior Engineering Competency Analyst with years of experience 
        helping engineers understand what it takes to advance in their careers. You excel at 
        parsing company leveling documents and translating them into actionable competency frameworks.
        
        Your goal is to analyze and define specific competency requirements for an engineer's 
        target promotion level based on company leveling documents."""
    
    def get_human_prompt_template(self) -> str:
        return """Analyze the competency requirements for promotion from {current_level} to {target_level} in {discipline}.

CONTEXT:
- Engineer Name: {name}
- Current Level: {current_level}
- Target Level: {target_level}
- Discipline: {discipline}

COMPANY LEVELING DOCUMENT:
{company_leveling_document}

YOUR TASK:
1. Parse and interpret the company leveling document
2. Identify technical, leadership, and soft skill requirements for the target level
3. Map discipline-specific expectations
4. Generate a comprehensive competency framework

OUTPUT FORMAT:
IMPORTANT: Return ONLY markdown text. Do NOT return JSON. Do NOT wrap your response in code blocks.

Provide a well-structured markdown document with the following sections:

## Competency Framework Overview
- **Engineer**: {name}
- **Current Level**: {current_level}
- **Target Level**: {target_level}
- **Discipline**: {discipline}

## Competency Categories

For each category, provide:
- **Category Name**: [Name]
- **Importance**: [High/Medium/Low]
- **Requirements**:
  - [List of specific requirements]
- **Evaluation Criteria**:
  - [How this will be assessed]

Categories to cover:
1. Technical Proficiency
2. Problem Solving
3. Impact & Scope
4. Leadership & Mentorship
5. Communication & Collaboration
6. Autonomy & Initiative
7. Business Acumen
8. Quality & Best Practices
9. Influence
10. Growth Mindset

## Level Differentiators
- **From {current_level}**: [What differentiates this level from the previous]
- **To {target_level}**: [What differentiates this level from the next]

## Expected Scope
[Describe the expected scope of work and responsibilities]

## Expected Impact
[Describe the expected impact and outcomes]

Be professional, objective, and encouraging. Use clear headings, bullet points, and structured formatting. Return pure markdown text only - no JSON, no code blocks."""
    
    def prepare_input(self, state: State) -> Dict[str, Any]:
        """Prepare input including company leveling document."""
        base_input = super().prepare_input(state)
        base_input["company_leveling_document"] = state["data_files"]["company_leveling_document"]
        return base_input
    
    def get_output_key(self) -> str:
        return "competency_analyzer_output"
    
    def execute(
        self,
        state: State,
        config: RunnableConfig | None = None
    ) -> Dict[str, Any]:
        """
        Execute competency analysis.
        
        Progress is shown via LangGraph's native events.
        """
        # Validate state
        validation_error = self.validate_state(state)
        if validation_error:
            return validation_error
        
        # Create LLM and prompt
        llm = self.create_llm()
        prompt = self.create_prompt()
        
        chain = prompt | llm
        
        # Prepare input data
        input_data = self.prepare_input(state)
        
        # Invoke chain (progress shown via LangGraph events)
        response = chain.invoke(input_data)
        
        # Extract content
        content = self.extract_response_content(response)
        
        # Post-process: Remove JSON code blocks if LLM still returns them
        # Strip ```json and ``` markers
        import re
        content = re.sub(r'^```json\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        # If content still looks like JSON (starts with {), try to extract markdown from it
        if content.startswith('{') and '"competency_categories"' in content:
            # This is JSON, we need to convert it or ask for regeneration
            # For now, wrap it in a note that it needs to be regenerated
            content = "**Note: This output was returned in JSON format. Please regenerate with markdown format.**\n\n" + content
        
        return {self.get_output_key(): content}


def competency_analyzer_node(
    state: Dict[str, Any],
    config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """
    Node function for LangGraph compatibility.
    
    Args:
        state: Current workflow state
        config: RunnableConfig for streaming
    
    Returns:
        Dictionary with competency_analyzer_output
    """
    agent = CompetencyAnalyzerAgent()
    return agent.execute(state, config)

