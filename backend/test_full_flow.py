#!/usr/bin/env python3
"""
Test the complete RAG flow that mirrors the actual application.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import config
from rag_system import RAGSystem

def main():
    """Test complete RAG flow"""
    print("=== Full RAG Flow Test ===")
    
    # Initialize RAG system
    print("1. Initializing RAG system...")
    rag_system = RAGSystem(config)
    
    # Test queries that should work
    test_queries = [
        "What is MCP?",
        "Tell me about course materials",
        "How do I use Anthropic's tools?",
        "What are the lessons about?",
        "Explain prompt compression"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        try:
            response, sources = rag_system.query(query)
            
            print(f"✓ Query successful")
            print(f"  Response length: {len(response)} chars")
            print(f"  Sources count: {len(sources)}")
            print(f"  Response preview: {response[:150]}...")
            
            if sources:
                print(f"  First source: {sources[0]}")
            
            # Check for error patterns
            if "query failed" in response.lower():
                print(f"✗ Response contains 'query failed'!")
            elif not response.strip():
                print(f"✗ Empty response!")
            elif response.startswith("I apologize") or response.startswith("I'm sorry"):
                print(f"⚠ Apologetic response - might indicate an issue")
            
        except Exception as e:
            print(f"✗ Query failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with different session scenarios
    print(f"\n6. Testing with session management...")
    try:
        session_id = "test_session"
        
        # First query in session
        response1, _ = rag_system.query("What is MCP?", session_id)
        print(f"✓ First query in session successful: {len(response1)} chars")
        
        # Follow-up query
        response2, _ = rag_system.query("Tell me more about that", session_id)
        print(f"✓ Follow-up query successful: {len(response2)} chars")
        
    except Exception as e:
        print(f"✗ Session test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test error scenarios
    print(f"\n7. Testing error scenarios...")
    try:
        # Very specific query that might not match
        response, sources = rag_system.query("What is quantum cryptography blockchain?")
        print(f"✓ Obscure query handled: {len(response)} chars")
        if "don't have information" in response.lower() or "not mentioned" in response.lower():
            print("  ✓ Correctly indicated no information available")
    except Exception as e:
        print(f"✗ Error scenario test failed: {e}")

if __name__ == "__main__":
    main()