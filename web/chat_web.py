# from flask import Flask, request, jsonify, render_template
# from flask_socketio import SocketIO, emit, send
# from glmtuner import ChatModel
# from glmtuner.tuner import get_infer_args
# import threading
# import queue
# import json
# import time

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)  # Initialize SocketIO with the app
# chat_model = ChatModel(*get_infer_args())
# history = {}
# history_lock = threading.Lock()

# request_queue = queue.Queue(maxsize=10)
# request_queue_lock = threading.Lock()

# @app.route('/',methods=['POST'])
# def chat():
#     data = request.get_json()
#     client_id = data.get('client_id', None)

#     with request_queue_lock:
#         if request_queue.full():
#             return jsonify({"response": "busy"})
#         request_queue.put((client_id, data))
#     print("client_id", data['client_id'], "query",data['query'])
#     socketio.emit('test_event_chat', {"message": "Test event after POST request!"})
#     print("test_event_chat emitted!")
#     return jsonify({"client_id": data['client_id'], "query": data['query']})

# def process_queue():
#     while True:
#         client_id, data = request_queue.get()

#         with app.app_context():  # 使用应用上下文
#             query = data.get('query', '')

#             if query.strip() == "clear":
#                 with history_lock:
#                     if client_id:
#                         client_history = history.get(client_id, [])
#                         client_history.clear()
#                 socketio.emit('chat_response', {"client_id": client_id, "response": "History has been removed."})
#                 #return jsonify({"response": "History has been removed."})
#                 continue

#             if query.strip() == "history":
#                 with history_lock:
#                     client_history = history.get(client_id, [])
#                     history_response = [{"client_id": client_id, "query": q, "response": a} for q, a in client_history]
#                 socketio.emit('chat_response', {"client_id": client_id, "response": history_response})
#                 #return jsonify({"history": history_response})
#                 continue

#             response = ""
#             with history_lock:
#                 client_history = history.setdefault(client_id, [])
#                 gen_kwargs = {
#                     "top_p": 0.4,
#                     "top_k": 0.3,
#                     "temperature": 0.95,
#                     "num_beams": 1,
#                     "max_length": 4096,
#                     "max_new_tokens": 1024,
#                     "repetition_penalty": 1.2,
#                 }
#                 print("print stream input")
#                 print(query)
#                 print(client_history)
#                 print(gen_kwargs)
#                 for new_text in chat_model.stream_chat(query, client_history, input_kwargs=gen_kwargs):
#                     response += new_text
#                 #time.sleep(10)  # to avoid hitting the API too hard
#                 # response = "the result without stream_chat"

#                 client_history.append((query, response))
#             print("About to emit chat_response...")
#             print(response)
#             #ocketio.emit('chat_response', {"client_id": client_id, "response": response})
#             socketio.emit('chat_response', {"client_id": client_id, "response": response}, namespace='/')  # 指定namespace
#             socketio.send({"client_id": client_id, "response": response}, namespace='/')
#             # return jsonify({"response": response})
#             print("chat_response emitted!")
#             request_queue.task_done()
#             # socketio.emit('test_event', {"message": "This is a test event from the server!"})
#             socketio.emit('test_event', {"message": "This is a test event from the server!"}, namespace='/')  # 指定namespace
#             print("test_event emitted!")

# @socketio.on('connect', namespace='/')
# def test_connect():
#     print('Client connected')

# @socketio.on('disconnect', namespace='/')
# def test_disconnect():
#     print('Client disconnected')

# # processing_thread = threading.Thread(target=process_queue)
# # processing_thread.start()
# # processing_thread.join()


# if __name__ == "__main__":
#     processing_thread = threading.Thread(target=process_queue)
#     processing_thread.start()
#     socketio.run(app, host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit, send
# from glmtuner import ChatModel
# from glmtuner.tuner import get_infer_args
# import eventlet  # Import eventlet
# import json

# eventlet.monkey_patch()  # Monkey patch standard libraries to use eventlet's cooperative multitasking

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, async_mode='eventlet')

# chat_model = ChatModel(*get_infer_args())
# history = {}
# history_lock = eventlet.semaphore.Semaphore()  # Use eventlet's Semaphore for locking

# request_queue = eventlet.queue.Queue(maxsize=10)  # Use eventlet's Queue

# @app.route('/', methods=['POST'])
# def chat():
#     data = request.get_json()
#     client_id = data.get('client_id', None)

#     if request_queue.full():
#         return jsonify({"response": "busy"})
#     request_queue.put((client_id, data))
#     print("client_id", data['client_id'], "query", data['query'])
#     socketio.emit('test_event_chat', {"message": "Test event after POST request!"})
#     print("test_event_chat emitted!")
#     return jsonify({"client_id": data['client_id'], "query": data['query']})

# def process_queue():
#     while True:
#         client_id, data = request_queue.get()

#         with app.app_context():
#             query = data.get('query', '')

#             if query.strip() == "clear":
#                 with history_lock:
#                     if client_id:
#                         client_history = history.get(client_id, [])
#                         client_history.clear()
#                 socketio.emit('chat_response', {"client_id": client_id, "response": "History has been removed."})
#                 #return jsonify({"response": "History has been removed."})
#                 continue

#             if query.strip() == "history":
#                 with history_lock:
#                     client_history = history.get(client_id, [])
#                     history_response = [{"client_id": client_id, "query": q, "response": a} for q, a in client_history]
#                 socketio.emit('chat_response', {"client_id": client_id, "response": history_response})
#                 #return jsonify({"history": history_response})
#                 continue

#             response = ""
#             with history_lock:
#                 client_history = history.setdefault(client_id, [])
#                 gen_kwargs = {
#                     "top_p": 0.4,
#                     "top_k": 0.3,
#                     "temperature": 0.95,
#                     "num_beams": 1,
#                     "max_length": 4096,
#                     "max_new_tokens": 1024,
#                     "repetition_penalty": 1.2,
#                 }
#                 print("print stream input")
#                 print(query)
#                 print(client_history)
#                 print(gen_kwargs)
#                 for new_text in chat_model.stream_chat(query, client_history, input_kwargs=gen_kwargs):
#                     response += new_text
#                 #time.sleep(10)  # to avoid hitting the API too hard
#                 # response = "the result without stream_chat"

#                 client_history.append((query, response))
#             print("About to emit chat_response...")
#             print(response)
#             #ocketio.emit('chat_response', {"client_id": client_id, "response": response})
#             socketio.emit('chat_response', {"client_id": client_id, "response": response}, namespace='/')  # 指定namespace
#             socketio.send({"client_id": client_id, "response": response}, namespace='/')
#             # return jsonify({"response": response})
#             print("chat_response emitted!")
#             request_queue.task_done()
#             # socketio.emit('test_event', {"message": "This is a test event from the server!"})
#             socketio.emit('test_event', {"message": "This is a test event from the server!"}, namespace='/')  # 指定namespace
#             print("test_event emitted!")

# @socketio.on('connect', namespace='/')
# def test_connect():
#     print('Client connected')

# @socketio.on('disconnect', namespace='/')
# def test_disconnect():
#     print('Client disconnected')

# if __name__ == "__main__":
#     socketio.start_background_task(process_queue)
#     socketio.run(app, host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit, send
# from glmtuner import ChatModel
# from glmtuner.tuner import get_infer_args
# import eventlet

# eventlet.monkey_patch()

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, async_mode='eventlet')

# chat_model = ChatModel(*get_infer_args())
# history = {}
# history_lock = eventlet.semaphore.Semaphore()

# request_queue = eventlet.queue.Queue(maxsize=10)

# @app.route('/', methods=['POST'])
# def chat():
#     data = request.get_json()
#     uid = data.get('uid', None)
#     qid = data.get('qid', None)
#     query = data.get('query', None)

#     if request_queue.full():
#         return jsonify({"uid": uid, "qid": qid, "response": "busy"})
#     request_queue.put((uid, qid, query))
#     print("uid", uid, "qid", qid, "query", query)
#     # socketio.emit('test_event_chat', {"message": "Test event after POST request!"})
#     # print("test_event_chat emitted!")
#     return jsonify({"uid": uid, "qid": qid, "query": query})

# def process_queue():
#     while True:
#         uid, qid, query = request_queue.get()

#         with app.app_context():
#             if query.strip() == "clear":
#                 with history_lock:
#                     if uid in history:
#                         del history[uid]
#                 socketio.emit('chat_response', {"qid": qid, "response": "History for UID {} has been removed.".format(uid)})
#                 continue

#             if query.strip() == "history":
#                 with history_lock:
#                     user_history = history.get(uid, [])
#                     # history_response = [{"uid": uid, "query": q, "response": a} for q, a in user_history]
#                     history_response = [{"query": q, "response": a} for q, a in user_history]
#                 socketio.emit('chat_response', {"qid": qid, "response": history_response})
#                 continue

#             response = ""
#             with history_lock:
#                 user_history = history.setdefault(uid, [])
#                 gen_kwargs = {
#                     "top_p": 0.4,
#                     "top_k": 0.3,
#                     "temperature": 0.95,
#                     "num_beams": 1,
#                     "max_length": 4096,
#                     "max_new_tokens": 1024,
#                     "repetition_penalty": 1.2,
#                 }
#                 for new_text in chat_model.stream_chat(query, user_history, input_kwargs=gen_kwargs):
#                     response += new_text

#                 user_history.append((query, response))
#             socketio.emit('chat_response', {"qid": qid, "response": response}, namespace='/')
#             print("chat_response emitted!")
#             request_queue.task_done()

# @socketio.on('connect', namespace='/')
# def test_connect():
#     print('Client connected')

# @socketio.on('disconnect', namespace='/')
# def test_disconnect():
#     print('Client disconnected')

# if __name__ == "__main__":
#     socketio.start_background_task(process_queue)
#     socketio.run(app, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify, make_response
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS  # 导入 CORS
from glmtuner import ChatModel
from glmtuner.tuner import get_infer_args
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
# CORS(app)  # 为你的 app 启用 CORS
# CORS(app, origins="*", methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"])  # 为你的 app 启用 CORS
CORS(app, origins='*', methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"], supports_credentials=True,expose_headers=['Origin', 'X-Requested-With', 'Content-Type', 'Accept']) 
socketio = SocketIO(app, cors_allowed_origins='*', logger=True, engineio_logger=True, async_mode='eventlet')

chat_model = ChatModel(*get_infer_args())
history = {}
history_lock = eventlet.semaphore.Semaphore()

request_queue = eventlet.queue.Queue(maxsize=10)

@app.route('/', methods=['POST'])
def chat():
    data = request.get_json()
    uid = data.get('uid', None)
    qid = data.get('qid', None)
    query = data.get('query', None)

    if request_queue.full():
        return jsonify({"uid": uid, "qid": qid, "response": "busy"})
    request_queue.put((uid, qid, query))
    print("uid", uid, "qid", qid, "query", query)
    # socketio.emit('test_event_chat', {"message": "Test event after POST request!"})
    # print("test_event_chat emitted!")
    return jsonify({"uid": uid, "qid": qid, "query": query})

def process_queue():
    while True:
        uid, qid, query = request_queue.get()

        with app.app_context():
            if query.strip() == "clear":
                with history_lock:
                    if uid in history:
                        del history[uid]
                socketio.emit('chat_response', {"qid": qid, "response": "History for UID {} has been removed.".format(uid)})
                continue

            if query.strip() == "history":
                with history_lock:
                    user_history = history.get(uid, [])
                    # history_response = [{"uid": uid, "query": q, "response": a} for q, a in user_history]
                    history_response = [{"query": q, "response": a} for q, a in user_history]
                socketio.emit('chat_response', {"qid": qid, "response": history_response})
                continue

            response = ""
            with history_lock:
                user_history = history.setdefault(uid, [])
                gen_kwargs = {
                    "top_p": 0.4,
                    "top_k": 0.3,
                    "temperature": 0.95,
                    "num_beams": 1,
                    "max_length": 4096,
                    "max_new_tokens": 1024,
                    "repetition_penalty": 1.2,
                }
                for new_text in chat_model.stream_chat(query, user_history, input_kwargs=gen_kwargs):
                    response += new_text

                user_history.append((query, response))
            socketio.emit('chat_response', {"qid": qid, "response": response}, namespace='/')
            print("chat_response emitted!")
            request_queue.task_done()

@socketio.on('connect', namespace='/')
def test_connect():
    print('Client connected')

@socketio.on('disconnect', namespace='/')
def test_disconnect():
    print('Client disconnected')

# @app.after_request
# def add_cors_headers(response):
#     # 允许所有来源
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     # 允许的HTTP方法
#     response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
#     # 允许的请求标头
#     response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept'
#     # 允许凭据（如果需要）
#     response.headers['Access-Control-Allow-Credentials'] = 'true'
#     # 预检请求的缓存时间（如果需要）
#     response.headers['Access-Control-Max-Age'] = '3600'
#     return response


if __name__ == "__main__":
    socketio.start_background_task(process_queue)
    socketio.run(app, host='0.0.0.0', port=5000)