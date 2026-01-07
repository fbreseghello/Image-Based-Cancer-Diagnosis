"""
Quick start script to run the Streamlit application.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app."""
    app_path = Path(__file__).parent / "src" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Starting Cancer Diagnosis AI Application")
    print("=" * 70)
    print(f"\nApp will open in your browser at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path)
        ])
    except KeyboardInterrupt:
        print("\n\nApplication stopped.")
    except Exception as e:
        print(f"\nError running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
