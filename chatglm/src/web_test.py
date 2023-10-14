from flask import Flask, request, jsonify, render_template
from glmtuner import ChatModel
from glmtuner.tuner import get_infer_args
import threading

app = Flask(__name__)
chat_model = ChatModel(*get_infer_args())
history = []
history_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '')

    # if query.strip() == "exit":
    #     return jsonify({"response": "Goodbye!"})

    if query.strip() == "clear":
        with history_lock:
            history.clear()
        # return jsonify({"response": "History has been removed."})

    if query.strip() == "":
        return jsonify({"response": ""})

    response = ""
    with history_lock:
        gen_kwargs = {
            "top_p": 0.4,
            "top_k": 0.3,
            "temperature": 0.95,
            "num_beams": 1,
            "max_length": 4096,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.2,
        }
        for new_text in chat_model.stream_chat(query, history, input_kwargs=gen_kwargs):
            response += new_text

        history.append((query, response))

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
