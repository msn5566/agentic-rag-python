#!/usr/bin/env python3
"""
Fix the Google Generative AI embedding model name format in config_google.json
"""

import json
import os

def fix_embedding_model_config():
    """Fix the embedding model name format in config_google.json"""
    config_path = "config_google.json"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using default config in app_google.py")
        return
    
    try:
        # Read current config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check if the embedding model needs fixing
        current_embed_model = config.get("google_embed", "")
        
        if current_embed_model == "text-embedding-004":
            print("Fixing embedding model name format...")
            config["google_embed"] = "models/text-embedding-004"
            
            # Write updated config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            
            print("SUCCESS: Successfully updated embedding model name to 'models/text-embedding-004'")
            print("The embedding model format has been fixed!")
            
        elif current_embed_model == "models/text-embedding-004":
            print("SUCCESS: Embedding model name is already in correct format")
            
        else:
            print(f"WARNING: Unknown embedding model format: {current_embed_model}")
            print("Please check your config_google.json file")
            
    except Exception as e:
        print(f"ERROR: Error fixing config: {str(e)}")
        print("Please manually update 'google_embed' to 'models/text-embedding-004' in config_google.json")

if __name__ == "__main__":
    fix_embedding_model_config()
