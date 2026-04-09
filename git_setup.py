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

def push_repo(token):
    print(f"Pushing to {remote_url}...")
    try:
        # Construct the URL with token for authentication
        # URL format: https://<token>@github.com/owner/repo.git
        auth_url = remote_url.replace("https://", f"https://{token}@")
        porcelain.push(repo_path, auth_url, b"refs/heads/master")
        print("Push successful!")
    except Exception as e:
        print(f"Push error: {e}")

if __name__ == "__main__":
    setup_repo()
    print("\nLocal repository is ready.")
    
    import sys
    if len(sys.argv) > 1:
        pat = sys.argv[1]
        push_repo(pat)
    else:
        print(f"To push to {remote_url}, please provide your GitHub Personal Access Token (PAT).")
        print("Usage: python git_setup.py <YOUR_PAT>")
