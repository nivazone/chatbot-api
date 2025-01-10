from app.blueprints.health import health_bp
from app.blueprints.chat import chat_bp

def register_routes(app):
    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp)