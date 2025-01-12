from flask import Flask
from app.config import Config
from app.routes import register_routes
from app.services.chat_service import initialize_chat_service;

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    with app.app_context():
        initialize_chat_service()
        
    register_routes(app)
    return app