#!/usr/bin/env python3
"""
Debug script to identify why queries are failing.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import config
from rag_system import RAGSystem

def main():
    """Debug the query system step by step"""
    print("=== RAG System Debug ===")
    
    # Check config
    print(f"ANTHROPIC_API_KEY: {'SET' if config.ANTHROPIC_API_KEY else 'NOT SET'}")
    print(f"CHROMA_PATH: {config.CHROMA_PATH}")
    print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
    
    # Initialize RAG system
    print("\n1. Initializing RAG System...")
    try:
        rag_system = RAGSystem(config)
        print("✓ RAG System initialized successfully")
    except Exception as e:
        print(f"✗ RAG System initialization failed: {e}")
        return
    
    # Check vector store data
    print("\n2. Checking vector store data...")
    try:
        course_count = rag_system.vector_store.get_course_count()
        print(f"✓ Found {course_count} courses")
        
        if course_count > 0:
            course_titles = rag_system.vector_store.get_existing_course_titles()
            print(f"  Courses: {course_titles}")
        else:
            print("✗ No courses found in vector store")
            return
    except Exception as e:
        print(f"✗ Vector store check failed: {e}")
        return
    
    # Test search tool directly
    print("\n3. Testing search tool directly...")
    try:
        search_result = rag_system.search_tool.execute("What is MCP?")
        print(f"✓ Search tool executed successfully")
        print(f"  Result length: {len(search_result)} characters")
        if search_result.startswith("Search error:"):
            print(f"✗ Search tool returned error: {search_result}")
        elif search_result.startswith("No relevant content found"):
            print(f"⚠ Search tool found no content: {search_result}")
        else:
            print(f"✓ Search tool returned content (first 200 chars): {search_result[:200]}...")
    except Exception as e:
        print(f"✗ Search tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test tool manager
    print("\n4. Testing tool manager...")
    try:
        tools = rag_system.tool_manager.get_tool_definitions()
        print(f"✓ Tool manager has {len(tools)} tools")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description'][:50]}...")
        
        # Test tool execution
        tool_result = rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="What is MCP?"
        )
        print(f"✓ Tool execution successful")
        print(f"  Result: {tool_result[:200]}...")
    except Exception as e:
        print(f"✗ Tool manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test AI generator (without actually calling Anthropic)
    print("\n5. Testing AI generator setup...")
    try:
        # Just test initialization and parameters
        print(f"✓ AI generator initialized with model: {rag_system.ai_generator.model}")
        print(f"  System prompt length: {len(rag_system.ai_generator.SYSTEM_PROMPT)} chars")
        print(f"  Has tools: {len(tools) > 0}")
    except Exception as e:
        print(f"✗ AI generator setup failed: {e}")
        return
    
    # Test full query without calling Anthropic
    print("\n6. Testing RAG system components...")
    try:
        # Test session manager
        session_id = rag_system.session_manager.create_session()
        print(f"✓ Session created: {session_id}")
        
        # Test that tools are available
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        print(f"✓ Tool definitions available: {len(tool_definitions)} tools")
        
        print("\n✓ All components are working. The issue might be:")
        print("  1. Anthropic API key invalid/expired")
        print("  2. Anthropic API call failing")
        print("  3. Network connectivity issues")
        print("  4. Rate limiting from Anthropic")
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()