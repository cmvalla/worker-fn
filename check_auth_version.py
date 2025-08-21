import google.auth
import inspect

def check_google_auth():
    """Prints diagnostic information about the google-auth library."""
    print("-" * 50)
    print("Google Auth Library Diagnostics")
    print("-" * 50)
    
    try:
        # Print version
        version = getattr(google.auth, '__version__', 'N/A')
        print(f"  Version: {version}")
        
        # Print file path
        path = inspect.getfile(google.auth)
        print(f"  Path: {path}")
        
        # Check for the problematic attribute
        has_attr = hasattr(google.auth, 'impersonated_credentials')
        print(f"  Has 'impersonated_credentials' attribute: {has_attr}")
        
        if not has_attr:
            print("[!] The 'impersonated_credentials' attribute is MISSING.")
        else:
            print("[*] The 'impersonated_credentials' attribute is PRESENT.")
            
    except Exception as e:
        print(f"[!] An error occurred during inspection: {e}")
        
    print("-" * 50)

if __name__ == "__main__":
    check_google_auth()
