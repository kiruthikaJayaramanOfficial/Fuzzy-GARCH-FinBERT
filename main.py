# Entry point for Streamlit Cloud
# This redirects to the actual app
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit",
                    "run", "apps/streamlit_app/app.py"])
