class SimpleAgent:
    def __init__(self):
        self.name = "SimpleAgent"
        self.system_prompt = """You are an AI assistant with access to 2 tools:
1. calculate(expression) - for math calculations
2. reverse_text(text) - for reversing text

Analyze the user's request and decide which tool to use. Respond with just the tool name and parameters.
Examples:
- "What is 5+3?" -> calculate(5+3)
- "Reverse the word hello" -> reverse_text(hello)
- "I need help with 10*2" -> calculate(10*2)"""
    
    def gpt_brain(self, user_input):
        """GPT-like decision making system"""
        # Simple pattern matching to simulate AI decision making
        user_lower = user_input.lower()
        
        # Check for math operations
        math_indicators = ['+', '-', '*', '/', 'calculate', 'math', 'add', 'subtract', 'multiply', 'divide', 'equals', '=']
        if any(indicator in user_lower for indicator in math_indicators):
            # Extract numbers and operators
            import re
            math_pattern = r'[\d+\-*/.() ]+'
            matches = re.findall(math_pattern, user_input)
            if matches:
                expression = ''.join(matches).strip()
                return f"calculate({expression})"
        
        # Check for reverse operations
        reverse_indicators = ['reverse', 'backwards', 'flip', 'invert']
        if any(indicator in user_lower for indicator in reverse_indicators):
            # Extract text to reverse
            words = user_input.split()
            text_words = [w for w in words if w.lower() not in reverse_indicators]
            text = ' '.join(text_words)
            return f"reverse_text({text})"
        
        # Default response
        return "I can help with calculations or reversing text. Please be more specific."
    
    def calculate(self, expression):
        """Tool 1: Simple calculator"""
        try:
            # Only allow basic math operations for safety
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Result: {result}"
            else:
                return "Error: Invalid characters in expression"
        except:
            return "Error: Invalid expression"
    
    def reverse_text(self, text):
        """Tool 2: Reverse text"""
        return f"Reversed: {text[::-1]}"
    
    def process_request(self, user_input):
        """Main processing logic with AI brain"""
        # Use AI brain to decide which tool to use
        decision = self.gpt_brain(user_input)
        
        # Parse the decision and execute the appropriate tool
        if decision.startswith("calculate("):
            expression = decision[10:-1]  # Remove "calculate(" and ")"
            return self.calculate(expression)
        
        elif decision.startswith("reverse_text("):
            text = decision[13:-1]  # Remove "reverse_text(" and ")"
            return self.reverse_text(text)
        
        else:
            return decision

def main():
    agent = SimpleAgent()
    print(f"Hello! I'm {agent.name}")
    print("I can help you with calculations and reversing text.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input:
            response = agent.process_request(user_input)
            print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()
