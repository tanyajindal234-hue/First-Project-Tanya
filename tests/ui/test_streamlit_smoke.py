import subprocess
import time
import requests

def test_streamlit_starts():
    """
    Smoke test to ensure the streamlit app can at least be parsed and started.
    We don't run a full browser test here, but check if the process stays alive.
    """
    process = subprocess.Popen(
        ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it a few seconds to start
    time.sleep(5)
    
    # Check if process is still running
    poll = process.poll()
    if poll is not None:
        stdout, stderr = process.communicate()
        print(f"Streamlit failed to start. Stdout: {stdout}, Stderr: {stderr}")
        assert False, "Streamlit process died early"
    
    # Terminate the process
    process.terminate()
    process.wait()
    assert True
