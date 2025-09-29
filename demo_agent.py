#!/usr/bin/env python3
"""
Demonstration of the Agentic Email Similarity Search System
Shows how the main agent uses tool calls to find similar emails
"""

import os
import json
from email_agent import EmailSimilarityAgent, analyze_email_similarity

def main():
    # Load environment variables
    with open('.env', 'r') as f:
        for line in f:
            if 'OPENAI_API_KEY' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
                break

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('OpenAI API key not found in .env file')
        return

    print("🤖 AGENTIC EMAIL SIMILARITY SEARCH SYSTEM")
    print("=" * 60)
    print("This demonstrates how the main agent uses tool calls to find similar emails")
    print()
    
    # Example 1: Basic agent usage
    print("📧 Example 1: Basic Agent Analysis")
    print("-" * 40)
    
    agent = EmailSimilarityAgent(api_key, email_dir='email_test')
    results = agent.analyze_email_and_find_similar('email_test/sample-580.eml')
    
    print("AGENT REASONING:")
    print(results['agent_reasoning'])
    print()
    
    print("SEARCH PARAMETERS:")
    print(json.dumps(results['search_parameters'], indent=2))
    print()
    
    print("SEARCH RESULTS:")
    if results['search_results']['success']:
        print(f"✅ Success: Found {len(results['search_results']['similar_emails'])} similar emails")
        print("\nInput Email:")
        input_email = results['search_results']['input_email']
        print(f"  File: {input_email['filename']}")
        print(f"  Subject: {input_email['subject']}")
        print(f"  Sender: {input_email['sender']}")
        
        print("\nSimilar Emails:")
        for i, email in enumerate(results['search_results']['similar_emails'][:3], 1):
            print(f"  {i}. {email['filename']} (Similarity: {email['similarity_score']})")
            print(f"     Subject: {email['subject']}")
            print(f"     Sender: {email['sender']}")
    else:
        print(f"❌ Failed: {results['search_results']['error']}")
    
    print("\nANALYSIS:")
    for insight in results['analysis']['insights']:
        print(f"  • {insight}")
    
    print("\nRECOMMENDATIONS:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    
    print("\nSUMMARY:")
    print(results['summary'])
    
    print("\n" + "=" * 60)
    
    # Example 2: Agent with specific requirements
    print("\n📧 Example 2: Agent with Specific Requirements")
    print("-" * 40)
    
    requirements = "Find emails with high similarity for phishing analysis"
    results2 = agent.analyze_email_and_find_similar(
        'email_test/sample-580.eml', 
        requirements
    )
    
    print("REQUIREMENTS:", requirements)
    print("ADJUSTED PARAMETERS:")
    print(json.dumps(results2['search_parameters'], indent=2))
    print(f"Found {len(results2['search_results']['similar_emails'])} emails with high similarity")
    
    print("\n" + "=" * 60)
    
    # Example 3: Convenience function usage
    print("\n📧 Example 3: Convenience Function")
    print("-" * 40)
    
    from email_agent import analyze_email_similarity
    results3 = analyze_email_similarity(
        'email_test/sample-580.eml',
        api_key,
        'email_test',
        "Find broad range of similar emails"
    )
    
    print("CONVENIENCE FUNCTION RESULTS:")
    print(f"Found {len(results3['search_results']['similar_emails'])} similar emails")
    print("Top 3 matches:")
    for i, email in enumerate(results3['search_results']['similar_emails'][:3], 1):
        print(f"  {i}. {email['filename']} - {email['similarity_score']}")
    
    print("\n✅ Agentic system demonstration complete!")
    print("\nKey Features Demonstrated:")
    print("• Agent reasoning about the task")
    print("• Dynamic parameter adjustment based on requirements")
    print("• Tool call execution for similarity search")
    print("• Result analysis and interpretation")
    print("• Intelligent recommendations")
    print("• Comprehensive summary generation")

if __name__ == "__main__":
    main()
