from huggingface_hub import login, HfApi
import os

# 1. Login
token = "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" # Aapka token
login(token=token)

api = HfApi()
repo_id = "jotaj30/quora-bert-model"

try:
    # 2. Pehle Repository banaiye (Agar pehle se bani hai toh skip ho jayega)
    print(f"Creating repository: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # 3. Ab Folder upload kijiye
    print("Model upload shuru ho raha hai... (418MB mein thoda time lagega)")
    
    # Check karein folder ka sahi naam kya hai (model ya quora_model)
    # Aapke error mein './quora_model' likha tha, verify kar lijiye
    api.upload_folder(
        folder_path="./quora_model",  # Agar folder ka naam 'model' hai toh yahi rehne dein
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\n✅ SUCCESS!  https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"\n❌ Error aaya: {e}")