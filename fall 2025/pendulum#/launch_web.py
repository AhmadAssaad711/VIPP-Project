"""
Launch the CartPole RL Web Visualizer
"""
import webbrowser
import time
import threading

def open_browser():
    """Open browser after short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:8000')

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 LAUNCHING CARTPOLE RL WEB VISUALIZER")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Interactive method selection (Policy Gradient, Q-Learning, LQR)")
    print("  ✓ Real-time visualization in browser")
    print("  ✓ Adjustable episodes and animation speed")
    print("  ✓ Live statistics and progress tracking")
    print("\nServer will start at: http://localhost:8000")
    print("Your browser will open automatically...")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the web server
    from web_app import app
    import uvicorn
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n\nServer stopped. Goodbye!")
