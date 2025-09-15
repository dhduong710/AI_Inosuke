import os
from flask import Flask, request, jsonify, render_template
from model_loader import chat_inosuke

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    avatar = data.get("avatar", "🧑")

    if not user_input:
        return jsonify({"response": "Nói gì đi chứ!!", "avatar": avatar})

    response = chat_inosuke(user_input)
    print(f"User: {user_input}")
    print(f"Inosuke: {response}")

    return jsonify({
        "response": response,
        "avatar": avatar
    })

if __name__ == "__main__":
    print("Templates folder:", app.template_folder)
    print("Static folder:", app.static_folder)
    app.run(host="0.0.0.0", port=7860, debug=False)
