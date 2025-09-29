"""
Main Agent for Email Similarity Search
This agent uses GPT-5 reasoning capabilities to analyze emails and find similar ones
"""

import os
import json
from typing import Dict, Any, List
from email_similarity_tool import initialize_email_similarity_tool, email_similarity_search

class EmailSimilarityAgent:
    """Main agent that orchestrates email similarity search using tool calls"""
    
    def __init__(self, api_key: str, email_dir: str = "email"):
        """
        Initialize the email similarity agent
        
        Args:
            api_key: OpenAI API key
            email_dir: Directory containing .eml files
        """
        self.api_key = api_key
        self.email_dir = email_dir
        
        # Initialize the email similarity tool
        self.tool = initialize_email_similarity_tool(api_key, email_dir)
        
        # Define available tools for the agent
        self.available_tools = {
            "email_similarity_search": {
                "description": "Find similar emails to a given email file",
                "parameters": {
                    "email_file": {
                        "type": "string",
                        "description": "Path to the input email file (.eml)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar emails to return (default: 10)",
                        "default": 10
                    },
                    "min_similarity": {
                        "type": "float",
                        "description": "Minimum similarity threshold (0.0-1.0, default: 0.0)",
                        "default": 0.7
                    }
                }
            }
        }
    
    def analyze_email_with_prompt(self, email_file: str, text_prompt: str) -> Dict[str, Any]:
        """
        Main agent method to analyze an email based on a text prompt
        
        Args:
            email_file: Path to the email file to analyze
            text_prompt: Text prompt describing what the user wants to do
            
        Returns:
            Dictionary containing analysis results based on the prompt
        """
        # Step 1: Agent reasoning about the task and prompt
        reasoning = self._reason_about_prompt(email_file, text_prompt)
        
        # Step 2: Determine if similarity search is needed
        needs_similarity_search = self._should_perform_similarity_search(text_prompt)
        
        if needs_similarity_search:
            # Step 3: Determine optimal parameters for similarity search
            search_params = self._determine_search_parameters(email_file, text_prompt)
            
            # Step 4: Execute the similarity search tool call
            search_results = self._execute_similarity_search(email_file, search_params)
            
            # Step 5: Analyze and interpret the results
            analysis = self._analyze_results(search_results, text_prompt)
            
            # Step 6: Provide final recommendations
            recommendations = self._provide_recommendations(search_results, analysis)
            
            return {
                "agent_reasoning": reasoning,
                "action_taken": "similarity_search",
                "search_parameters": search_params,
                "search_results": search_results,
                "analysis": analysis,
                "recommendations": recommendations,
                "summary": self._generate_summary(search_results, analysis)
            }
        else:
            # No similarity search needed - return message about missing functionality
            return {
                "agent_reasoning": reasoning,
                "action_taken": "no_action",
                "message": "Tool call for this request has not been added yet",
                "requested_action": self._extract_requested_action(text_prompt),
                "available_tools": ["email_similarity_search"]
            }
    
    def _reason_about_prompt(self, email_file: str, text_prompt: str) -> str:
        """Agent reasoning about the user's text prompt"""
        reasoning = f"""
        AGENT REASONING:
        
        Email File: {email_file}
        User Prompt: "{text_prompt}"
        
        Analysis:
        1. I need to understand what the user wants to do with this email
        2. Determine if the request involves finding similar emails
        3. If similarity search is needed, I'll use the email_similarity_search tool
        4. If other functionality is requested, I'll inform that the tool is not available yet
        
        Strategy:
        - Parse the user's intent from the text prompt
        - Look for keywords indicating similarity search requests
        - Make intelligent decision about tool usage
        """
        
        return reasoning.strip()
    
    def _should_perform_similarity_search(self, text_prompt: str) -> bool:
        """
        Intelligently determine if the text prompt requires similarity search
        
        Args:
            text_prompt: User's text prompt
            
        Returns:
            True if similarity search is needed, False otherwise
        """
        # Convert to lowercase for case-insensitive analysis
        prompt_lower = text_prompt.lower()
        
        # Keywords that indicate similarity search is needed
        similarity_keywords = [
            "similar", "similar emails", "find similar", "look for similar",
            "match", "matching", "matches", "find matches",
            "related", "related emails", "find related",
            "like this", "like this email", "emails like this",
            "compare", "comparison", "duplicate", "duplicates",
            "same", "same type", "same kind", "resembl", "pattern"
        ]
        
        # Keywords that indicate other actions (not similarity search)
        other_action_keywords = [
            "classify", "classification", "categorize", "category",
            "extract", "extraction", "parse", "parsing",
            "analyze content", "content analysis", "sentiment",
            "translate", "translation", "summarize", "summary",
            "validate", "validation", "check format", "verify"
        ]
        
        # Check for similarity keywords
        for keyword in similarity_keywords:
            if keyword in prompt_lower:
                return True
        
        # Check for explicit other actions that would override similarity search
        for keyword in other_action_keywords:
            if keyword in prompt_lower:
                return False
        
        # If prompt contains question words with email context, likely similarity search
        question_patterns = [
            "what emails are", "which emails", "show me emails",
            "find emails", "get emails", "emails that are",
            "other emails", "more emails"
        ]
        
        for pattern in question_patterns:
            if pattern in prompt_lower:
                return True
        
        # Default: if unclear, assume they want similarity search for email analysis
        return True
    
    def _extract_requested_action(self, text_prompt: str) -> str:
        """
        Extract what action the user is requesting from the prompt
        
        Args:
            text_prompt: User's text prompt
            
        Returns:
            Description of the requested action
        """
        prompt_lower = text_prompt.lower()
        
        # Map keywords to actions
        action_mapping = {
            "classify": "email classification",
            "categorize": "email categorization", 
            "extract": "content extraction",
            "parse": "email parsing",
            "analyze content": "content analysis",
            "sentiment": "sentiment analysis",
            "translate": "translation",
            "summarize": "email summarization",
            "validate": "email validation",
            "verify": "email verification"
        }
        
        for keyword, action in action_mapping.items():
            if keyword in prompt_lower:
                return action
        
        # If no specific action identified, return general description
        return "general email analysis"
    
    def _reason_about_task(self, email_file: str, requirements: str = None) -> str:
        """Agent reasoning about the email similarity task (legacy method)"""
        reasoning = f"""
        AGENT REASONING:
        
        Task: Find similar emails to {email_file}
        
        Analysis:
        1. I need to analyze the input email to understand its content and characteristics
        2. Based on the email content, I should determine appropriate similarity search parameters
        3. I'll use the email_similarity_search tool to find similar emails
        4. I'll analyze the results to provide meaningful insights
        
        Strategy:
        - Use a moderate similarity threshold (0.5-0.7) for balanced results
        - Return top 10 similar emails for comprehensive analysis
        - Focus on semantic similarity rather than exact matches
        """
        
        if requirements:
            reasoning += f"\n\nSpecific Requirements: {requirements}"
        
        return reasoning.strip()
    
    def _determine_search_parameters(self, email_file: str, requirements: str = None) -> Dict[str, Any]:
        """Determine optimal search parameters based on the task"""
        # Default parameters
        params = {
            "top_k": 10,
            "min_similarity": 0.5  # Moderate threshold for balanced results
        }
        
        # Adjust parameters based on requirements
        if requirements:
            if "high similarity" in requirements.lower() or "strict" in requirements.lower():
                params["min_similarity"] = 0.7
            elif "loose" in requirements.lower() or "broad" in requirements.lower():
                params["min_similarity"] = 0.3
            elif "many results" in requirements.lower():
                params["top_k"] = 15
            elif "few results" in requirements.lower():
                params["top_k"] = 5
        
        return params
    
    def _execute_similarity_search(self, email_file: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the similarity search tool call"""
        try:
            results = email_similarity_search(
                email_file=email_file,
                top_k=params["top_k"],
                min_similarity=params["min_similarity"]
            )
            return results
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "input_email": {"filename": email_file},
                "similar_emails": []
            }
    
    def _analyze_results(self, results: Dict[str, Any], requirements: str = None) -> Dict[str, Any]:
        """Analyze the similarity search results"""
        if not results.get("success", False):
            return {
                "status": "failed",
                "error": results.get("error", "Unknown error"),
                "insights": []
            }
        
        similar_emails = results.get("similar_emails", [])
        input_email = results.get("input_email", {})
        
        # Analyze similarity patterns
        insights = []
        
        if similar_emails:
            # Calculate similarity statistics
            similarities = [email["similarity_score"] for email in similar_emails]
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            
            insights.append(f"Found {len(similar_emails)} similar emails")
            insights.append(f"Average similarity: {avg_similarity:.3f}")
            insights.append(f"Similarity range: {min_similarity:.3f} - {max_similarity:.3f}")
            
            # Analyze content patterns
            subjects = [email["subject"] for email in similar_emails if email["subject"]]
            if subjects:
                insights.append(f"Most similar emails have subjects related to: {subjects[0][:50]}...")
            
            # Analyze sender patterns
            senders = [email["sender"] for email in similar_emails if email["sender"]]
            unique_senders = len(set(senders))
            insights.append(f"Similar emails come from {unique_senders} different senders")
            
        else:
            insights.append("No similar emails found with the current threshold")
        
        return {
            "status": "success",
            "insights": insights,
            "statistics": {
                "total_found": results.get("total_found", 0),
                "returned": results.get("returned", 0),
                "threshold_used": results.get("min_similarity_threshold", 0.0)
            }
        }
    
    def _provide_recommendations(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Provide recommendations based on the analysis"""
        recommendations = []
        
        if not results.get("success", False):
            recommendations.append("Check if the email file exists and contains readable content")
            return recommendations
        
        similar_emails = results.get("similar_emails", [])
        
        if not similar_emails:
            recommendations.append("Consider lowering the similarity threshold to find more results")
            recommendations.append("The email might be unique or very different from others in the collection")
        else:
            recommendations.append(f"Found {len(similar_emails)} similar emails with good similarity scores")
            
            # Check if we should adjust threshold
            similarities = [email["similarity_score"] for email in similar_emails]
            if similarities and similarities[0] > 0.8:
                recommendations.append("High similarity found - these emails are very similar")
            elif similarities and similarities[0] < 0.6:
                recommendations.append("Consider raising the threshold for more precise matches")
            
            # Content-based recommendations
            if len(similar_emails) >= 5:
                recommendations.append("Multiple similar emails found - good for pattern analysis")
            elif len(similar_emails) < 3:
                recommendations.append("Few similar emails found - consider broader search criteria")
        
        return recommendations
    
    def _generate_summary(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate a summary of the entire analysis"""
        if not results.get("success", False):
            return f"Analysis failed: {results.get('error', 'Unknown error')}"
        
        input_email = results.get("input_email", {})
        similar_emails = results.get("similar_emails", [])
        
        summary = f"""
        EMAIL SIMILARITY ANALYSIS SUMMARY
        
        Input Email: {input_email.get('filename', 'Unknown')}
        Subject: {input_email.get('subject', 'No subject')}
        Sender: {input_email.get('sender', 'Unknown sender')}
        
        Results: Found {len(similar_emails)} similar emails
        """
        
        if similar_emails:
            top_match = similar_emails[0]
            summary += f"""
        Top Match: {top_match['filename']} (Similarity: {top_match['similarity_score']})
        Top Match Subject: {top_match['subject']}
        """
        
        insights = analysis.get("insights", [])
        if insights:
            summary += "\nKey Insights:\n"
            for insight in insights:
                summary += f"- {insight}\n"
        
        return summary.strip()
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available tools"""
        return self.available_tools

# Convenience functions for direct usage
def analyze_email_with_prompt(email_file: str, text_prompt: str, api_key: str, email_dir: str = "email") -> Dict[str, Any]:
    """
    Convenience function to analyze email with a text prompt using the agent
    
    Args:
        email_file: Path to the email file to analyze
        text_prompt: Text prompt describing what the user wants to do
        api_key: OpenAI API key
        email_dir: Directory containing .eml files
        
    Returns:
        Dictionary containing complete analysis results
    """
    agent = EmailSimilarityAgent(api_key, email_dir)
    return agent.analyze_email_with_prompt(email_file, text_prompt)

def analyze_email_similarity(email_file: str, api_key: str, email_dir: str = "email", requirements: str = None) -> Dict[str, Any]:
    """
    Legacy convenience function for backward compatibility
    
    Args:
        email_file: Path to the email file to analyze
        api_key: OpenAI API key
        email_dir: Directory containing .eml files
        requirements: Optional specific requirements for the analysis
        
    Returns:
        Dictionary containing complete analysis results
    """
    # Convert requirements to a similarity search prompt
    if requirements:
        text_prompt = f"Find similar emails: {requirements}"
    else:
        text_prompt = "Find similar emails to this one"
    
    agent = EmailSimilarityAgent(api_key, email_dir)
    return agent.analyze_email_with_prompt(email_file, text_prompt)
