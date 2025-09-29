#!/usr/bin/env python3
"""
Demonstration script for enhanced email similarity search with detailed logging
"""

import os
import sys
sys.path.append('email')

from similar_emails import EmailSimilaritySearch

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

    print("🔍 Enhanced Email Similarity Search with Detailed Logging")
    print("=" * 60)
    
    # Initialize searcher
    searcher = EmailSimilaritySearch(api_key, email_dir='email_test')
    
    # Example 1: Find similar emails with automatic logging
    print("\n📧 Example 1: Automatic Logging During Search")
    print("-" * 40)
    results = searcher.find_similar_emails('email_test/sample-580.eml', top_k=2, min_similarity=0.75)
    
    # Example 2: Manual logging of specific emails
    print("\n📧 Example 2: Manual Email Content Logging")
    print("-" * 40)
    
    # Log the input email in detail
    input_email = searcher.emails_data['sample-580.eml']
    print("Input Email Details:")
    searcher.log_email_content(input_email, show_full_content=False)
    
    # Log the most similar email in detail
    if results:
        most_similar = results[0][2]  # Get EmailData from tuple
        print("Most Similar Email Details:")
        searcher.log_email_content(most_similar, show_full_content=False)
    
    # Example 3: Show full content for debugging
    print("\n📧 Example 3: Full Content Logging (for debugging)")
    print("-" * 40)
    print("Full content of input email:")
    searcher.log_email_content(input_email, show_full_content=True)
    
    print("\n✅ Logging demonstration complete!")
    print("The enhanced logging shows:")
    print("• Input email subject, sender, and content preview")
    print("• Similar emails found with similarity scores")
    print("• Detailed content analysis for debugging")
    print("• Full content logging when needed")

if __name__ == "__main__":
    main()
