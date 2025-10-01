#!/usr/bin/env python3
"""
Simple interface to run the agentic email analysis with text prompts
"""
# TODO: actual RAG database, chroma DB, processing should appear

import os
import sys
from email_agent import analyze_email_with_prompt

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_agent.py <email_file> <text_prompt>")
        print("Example: python run_agent.py email_test/sample-580.eml 'Find similar emails to this one'")
        print("Example: python run_agent.py email_test/sample-580.eml 'Classify this email'")
        print("Example: python run_agent.py email_test/sample-580.eml 'Extract the main content'")
        return
    
    email_file = sys.argv[1]
    text_prompt = sys.argv[2]
    
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
        print('❌ OpenAI API key not found in .env file')
        return
    
    print(f"🤖 Analyzing email: {email_file}")
    print(f"💬 User prompt: {text_prompt}")
    print()
    
    try:
        # Run the agentic analysis with prompt
        results = analyze_email_with_prompt(
            email_file=email_file,
            text_prompt=text_prompt,
            api_key=api_key,
            email_dir='email_test'
        )
        
        # Display results based on action taken
        print("🎯 ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"🧠 Agent Reasoning:")
        print(results['agent_reasoning'])
        print()
        
        print(f"⚡ Action Taken: {results['action_taken']}")
        
        if results['action_taken'] == 'similarity_search':
            if results['search_results']['success']:
                similar_emails = results['search_results']['similar_emails']
                print(f"✅ Found {len(similar_emails)} similar emails")
                
                print(f"\n📧 INPUT EMAIL:")
                input_email = results['search_results']['input_email']
                print(f"   File: {input_email['filename']}")
                print(f"   Subject: {input_email['subject']}")
                print(f"   Sender: {input_email['sender']}")
                
                print(f"\n🔍 SIMILAR EMAILS:")
                for i, email in enumerate(similar_emails, 1):
                    print(f"   {i}. {email['filename']} (Similarity: {email['similarity_score']})")
                    print(f"      Subject: {email['subject']}")
                    print(f"      Sender: {email['sender']}")
                    print()
                
                print("💡 INSIGHTS:")
                for insight in results['analysis']['insights']:
                    print(f"   • {insight}")
                
                print("\n🎯 RECOMMENDATIONS:")
                for rec in results['recommendations']:
                    print(f"   • {rec}")
                    
                print("\n📝 SUMMARY:")
                print(results['summary'])
            else:
                print(f"❌ Similarity search failed: {results['search_results']['error']}")
        
        elif results['action_taken'] == 'no_action':
            print(f"\n💬 {results['message']}")
            print(f"🎯 Requested Action: {results['requested_action']}")
            print(f"🛠️  Available Tools: {', '.join(results['available_tools'])}")
            print("\n💡 To find similar emails, try prompts like:")
            print("   • 'Find similar emails to this one'")
            print("   • 'Show me emails like this'")
            print("   • 'Find matching emails'")
            print("   • 'Look for related emails'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
