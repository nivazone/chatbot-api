import os
from dotenv import load_dotenv
from app import create_app

load_dotenv()

def main():
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()