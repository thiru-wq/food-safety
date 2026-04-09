import os
from dulwich import porcelain

repo_path = r'.'
remote_url = 'https://github.com/thiru-wq/food-safety.git'

def setup_repo():
    if not os.path.exists('.git'):
        print("Initializing repository...")
        porcelain.init(repo_path)
    
    print("Staging files...")
    # Porcelain add handles .gitignore automatically? 
    # Let's be manual to be safe
    files_to_add = []
    for root, dirs, files in os.walk(repo_path):
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        if 'models' in dirs:
            dirs.remove('models')
        if 'dataset' in dirs:
            dirs.remove('dataset')
            
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), repo_path)
            files_to_add.append(rel_path)
            
    porcelain.add(repo_path, files_to_add)
    
    try:
        print("Committing...")
        porcelain.commit(repo_path, b"Initial commit from Antigravity AI")
    except Exception as e:
        print(f"Commit error (maybe nothing to commit): {e}")

if __name__ == "__main__":
    setup_repo()
    print("\nLocal repository is ready.")
    print(f"To push to {remote_url}, please provide a GitHub Personal Access Token (PAT).")
    print("I can then attempt the push for you.")
