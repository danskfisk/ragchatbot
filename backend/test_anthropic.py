#!/usr/bin/env python3
"""
Test Anthropic API connectivity.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import config
import anthropic

def main():
    """Test Anthropic API"""
    print("=== Anthropic API Test ===")
    
    if not config.ANTHROPIC_API_KEY:
        print("✗ No API key found")
        return
    
    print(f"API Key: {config.ANTHROPIC_API_KEY[:20]}...")
    print(f"Model: {config.ANTHROPIC_MODEL}")
    
    # Test basic API call
    print("\n1. Testing basic API connection...")
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        print("✓ Client created successfully")
        
        # Simple test message
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        
        print("✓ API call successful")
        print(f"Response: {response.content[0].text}")
        
    except anthropic.AuthenticationError:
        print("✗ Authentication failed - API key is invalid or expired")
    except anthropic.PermissionDeniedError:
        print("✗ Permission denied - API key doesn't have access to this model")
    except anthropic.RateLimitError:
        print("✗ Rate limit exceeded")
    except anthropic.APIConnectionError:
        print("✗ Network connection failed")
    except Exception as e:
        print(f"✗ API call failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()