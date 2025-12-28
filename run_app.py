import os
import sys
import time
import webbrowser
import socket
import threading

# Where to store persistent data:
# - If running as EXE (frozen), use the EXE folder
# - Otherwise use the script folder
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

PORT = 8501
URL = f"http://127.0.0.1:{PORT}"

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, "uploaded_files"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "chroma_db"), exist_ok=True)

def port_is_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def open_browser_once():
    flag = os.path.join(BASE_DIR, ".opened_browser")
    if not os.path.exists(flag):
        webbrowser.open(URL)
        try:
            with open(flag, "w", encoding="utf-8") as f:
                f.write("1")
        except Exception:
            pass

def run_streamlit_in_process():
    from streamlit.web import cli as stcli

    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(bundle_dir, "app.py")

    # IMPORTANT: set development mode to false, otherwise server.port can error
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.address=127.0.0.1",
        f"--server.port={PORT}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
        "--browser.gatherUsageStats=false",
    ]

    def waiter():
        for _ in range(250):
            if port_is_open("127.0.0.1", PORT):
                open_browser_once()
                return
            time.sleep(0.1)

    threading.Thread(target=waiter, daemon=True).start()

    stcli.main()

if __name__ == "__main__":
    run_streamlit_in_process()
