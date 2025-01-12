from flask import Blueprint, request, jsonify
from app.services.chat_service import run_workflow
from langchain_core.messages import HumanMessage

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
async def chat():
    data = request.json
    query = data.get("query")
    language = data.get("language", "English")
    chat_id = data.get("chat_id")

    if not query:
        return jsonify({"error": "The 'query' parameter is required"}), 400
    if not chat_id:
        return jsonify({"error": "The 'chat_id' parameter is required"}), 400

    
    input_messages = [HumanMessage(content=query)]
    
    try:
        response = await run_workflow(input_messages, language, chat_id)
        return jsonify({"chat_id": chat_id, "response": response.content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500