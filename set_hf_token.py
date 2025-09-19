#!/usr/bin/env python3
"""
Manual HuggingFace token setup for WSL environments.
"""

import os
from huggingface_hub import login, whoami

def set_token_manually():
    """
    Set HuggingFace token manually.
    """
    print("🔧 Manual HuggingFace Token Setup")
    print("=" * 40)
    
    # Get token from user input (visible in terminal)
    print("Please paste your HuggingFace token below:")
    print("(Token will be visible - make sure you're in a secure environment)")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    try:
        # Login with the token
        print("🔄 Authenticating...")
        login(token=token, add_to_git_credential=True)
        
        # Verify it worked
        user_info = whoami()
        print(f"✅ Successfully authenticated as: {user_info.get('name', 'Unknown')}")
        print("🎉 Token saved for future use!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        return False

def test_access():
    """
    Test EmbeddingGemma access.
    """
    try:
        print("\n🧪 Testing EmbeddingGemma access...")
        from sentence_transformers import SentenceTransformer
        
        # Just check if we can access the model metadata (don't download yet)
        model = SentenceTransformer("google/embeddinggemma-300m", trust_remote_code=True)
        print("✅ EmbeddingGemma is accessible!")
        return True
        
    except Exception as e:
        print(f"❌ Cannot access EmbeddingGemma: {str(e)}")
        if "gated" in str(e).lower():
            print("💡 Make sure you've accepted the license at:")
            print("   https://huggingface.co/google/embeddinggemma-300m")
        return False

if __name__ == "__main__":
    print("📋 Steps to complete:")
    print("1. Get token from: https://huggingface.co/settings/tokens")
    print("2. Accept license: https://huggingface.co/google/embeddinggemma-300m")
    print("3. Run this script with your token")
    print()
    
    if set_token_manually():
        test_access()
    else:
        print("❌ Setup failed. Please check your token and try again.")
