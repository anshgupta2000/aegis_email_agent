# Agentic Email Similarity Search System

A sophisticated agentic system that uses GPT-5 reasoning capabilities to analyze emails and find similar ones through intelligent tool calls.

## 🤖 System Architecture

### Main Components

1. **Main Agent (`EmailSimilarityAgent`)**: GPT-5 powered reasoning agent
2. **Tool Interface (`EmailSimilarityTool`)**: Specialized tool for email similarity search
3. **Tool Call Function (`email_similarity_search`)**: Executable function for agent calls

### Agentic Flow

```
Input Email → Agent Reasoning → Parameter Determination → Tool Call → Result Analysis → Recommendations
```

## 🚀 Features

- **Intelligent Reasoning**: Agent analyzes the task and determines optimal parameters
- **Dynamic Parameter Adjustment**: Adjusts similarity thresholds based on requirements
- **Tool Call Interface**: Clean separation between agent logic and tool execution
- **Comprehensive Analysis**: Provides insights, recommendations, and summaries
- **Flexible Requirements**: Supports custom analysis requirements

## 📁 File Structure

```
phishing_pot/
├── email_agent.py              # Main agent with GPT-5 reasoning
├── email_similarity_tool.py     # Tool implementation for similarity search
├── run_agent.py                 # Simple command-line interface
├── demo_agent.py                # Comprehensive demonstration
└── AGENTIC_SYSTEM_README.md     # This documentation
```

## 🛠️ Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 🎯 Usage

### Command Line Interface

```bash
# Basic usage
python run_agent.py email_test/sample-580.eml

# With specific requirements
python run_agent.py email_test/sample-580.eml "Find high similarity emails for phishing analysis"
```

### Programmatic Usage

```python
from email_agent import EmailSimilarityAgent, analyze_email_similarity

# Initialize agent
agent = EmailSimilarityAgent(api_key="your-key", email_dir="email_test")

# Analyze email with agent reasoning
results = agent.analyze_email_and_find_similar(
    email_file="email_test/sample-580.eml",
    requirements="Find emails with high similarity for phishing analysis"
)

# Or use convenience function
results = analyze_email_similarity(
    email_file="email_test/sample-580.eml",
    api_key="your-key",
    email_dir="email_test",
    requirements="Find broad range of similar emails"
)
```

## 🧠 Agent Reasoning Process

### 1. Task Analysis
The agent analyzes the input email and requirements to understand:
- What type of similarity search is needed
- What parameters would be optimal
- What insights to focus on

### 2. Parameter Determination
Based on requirements, the agent adjusts:
- **Similarity Threshold**: 
  - "high similarity" → 0.7+
  - "broad range" → 0.3+
  - Default → 0.5
- **Number of Results**: 
  - "many results" → 15
  - "few results" → 5
  - Default → 10

### 3. Tool Call Execution
The agent calls the `email_similarity_search` tool with determined parameters.

### 4. Result Analysis
The agent analyzes the results to provide:
- Similarity statistics
- Content pattern insights
- Sender analysis
- Quality assessment

### 5. Recommendations
The agent provides intelligent recommendations based on the analysis.

## 📊 Example Output

```
🤖 Analyzing email: email_test/sample-580.eml
📋 Requirements: Find emails with high similarity for phishing analysis

🎯 ANALYSIS RESULTS
==================================================
✅ Found 8 similar emails

📧 INPUT EMAIL:
   File: sample-580.eml
   Subject: Fw: Good evening:) my Amazing.
   Sender: phishing@pot <phishing@pot>

🔍 SIMILAR EMAILS:
   1. sample-1871.eml (Similarity: 0.7791)
      Subject: phishing@pot, Please verify
      Sender: SignalMax HD TV Antenna - Free Trial

💡 INSIGHTS:
   • Found 8 similar emails
   • Average similarity: 0.752
   • Similarity range: 0.729 - 0.779
   • Most similar emails have subjects related to: phishing@pot, Please verify...
   • Similar emails come from 8 different senders

🎯 RECOMMENDATIONS:
   • Found 8 similar emails with good similarity scores
   • Multiple similar emails found - good for pattern analysis

📝 SUMMARY:
EMAIL SIMILARITY ANALYSIS SUMMARY
        
        Input Email: sample-580.eml
        Subject: Fw: Good evening:) my Amazing.
        Sender: phishing@pot <phishing@pot>
        
        Results: Found 8 similar emails
        
        Top Match: sample-1871.eml (Similarity: 0.7791)
        Top Match Subject: phishing@pot, Please verify
```

## 🔧 Tool Interface

### Available Tools

```python
{
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
                "default": 0.0
            }
        }
    }
}
```

### Tool Call Function

```python
def email_similarity_search(email_file: str, top_k: int = 10, min_similarity: float = 0.0) -> Dict:
    """
    Tool call function for email similarity search
    
    Returns:
        Dictionary with search results including:
        - success: bool
        - input_email: dict with email details
        - similar_emails: list of similar emails
        - total_found: int
        - returned: int
        - min_similarity_threshold: float
    """
```

## 🎛️ Configuration

### Agent Parameters

- **Default Similarity Threshold**: 0.5 (moderate matching)
- **Default Top K**: 10 emails
- **Requirements Processing**: Automatic parameter adjustment

### Tool Parameters

- **Content Length Limit**: 4000 characters
- **Embedding Model**: text-embedding-ada-002
- **Caching**: Automatic embeddings cache

## 🔍 Advanced Usage

### Custom Requirements

```python
# High precision analysis
results = agent.analyze_email_and_find_similar(
    email_file="email_test/sample-580.eml",
    requirements="Find emails with high similarity for phishing analysis"
)

# Broad analysis
results = agent.analyze_email_and_find_similar(
    email_file="email_test/sample-580.eml",
    requirements="Find broad range of similar emails for research"
)
```

### Direct Tool Usage

```python
from email_similarity_tool import initialize_email_similarity_tool, email_similarity_search

# Initialize tool
tool = initialize_email_similarity_tool(api_key, email_dir)

# Direct tool call
results = email_similarity_search(
    email_file="email_test/sample-580.eml",
    top_k=15,
    min_similarity=0.7
)
```

## 🧪 Testing

Run the demonstration:
```bash
python demo_agent.py
```

Test with different requirements:
```bash
python run_agent.py email_test/sample-580.eml "Find high similarity emails"
python run_agent.py email_test/sample-580.eml "Find broad range of similar emails"
python run_agent.py email_test/sample-580.eml "Find few precise matches"
```

## 🎯 Benefits of Agentic Approach

1. **Intelligent Reasoning**: Agent understands context and requirements
2. **Dynamic Adaptation**: Automatically adjusts parameters based on needs
3. **Comprehensive Analysis**: Provides insights beyond simple similarity scores
4. **Tool Separation**: Clean separation between reasoning and execution
5. **Extensible**: Easy to add new tools and capabilities
6. **User-Friendly**: Natural language requirements processing

## 🔮 Future Enhancements

- **Multi-tool Integration**: Add more specialized tools
- **Advanced Reasoning**: More sophisticated parameter optimization
- **Interactive Mode**: Real-time conversation with the agent
- **Batch Processing**: Analyze multiple emails simultaneously
- **Custom Analysis**: Domain-specific analysis capabilities

## 📝 License

This implementation follows the same license as the parent project.
