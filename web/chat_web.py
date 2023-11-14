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

from flask import Flask, request, jsonify, make_response, has_app_context
from flask_socketio import SocketIO, emit, send, join_room, leave_room
from flask_cors import CORS  # 导入 CORS
from glmtuner import ChatModel
from glmtuner.tuner import get_infer_args
from flask_sqlalchemy import SQLAlchemy
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
# CORS(app)  # 为你的 app 启用 CORS
# CORS(app, origins="*", methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"])  # 为你的 app 启用 CORS
CORS(app, origins='*', methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"], supports_credentials=True,expose_headers=['Origin', 'X-Requested-With', 'Content-Type', 'Accept']) 
socketio = SocketIO(app, cors_allowed_origins='*', logger=True, engineio_logger=True, async_mode='eventlet')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///AmovAi_database.db'
db = SQLAlchemy(app)

chat_model = ChatModel(*get_infer_args())
history = {}
history_lock = eventlet.semaphore.Semaphore()
connected_users_sid_lock = eventlet.semaphore.Semaphore()

request_queue = eventlet.queue.Queue(maxsize=10)
connected_users = {}
connected_users_sid = {}

# 模拟的数据库模型，实际项目中应该定义真正的字段和关系
class QueryResponse(db.Model):
    __tablename__ = 'query_response'
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(255))
    qid = db.Column(db.String(255), nullable=False)
    sub_qid = db.Column(db.String(255), nullable=False)  # 新增字段
    Query = db.Column(db.String(4096), nullable=False)
    response = db.Column(db.String(4096), nullable=False)

    def __init__(self, uid, qid, sub_qid, Query, response):
        self.uid = uid
        self.qid = qid
        self.sub_qid = sub_qid
        self.Query = Query
        self.response = response

# db.create_all()  # 创建数据库表
# db.session.commit()

def safe_commit():
    try:
        if has_app_context():
            db.session.commit()
        else:
            with app.app_context():
                db.session.commit()
    except Exception as e:
        db.session.rollback()  # 在异常情况下回滚事务
        print(f"An error occurred while committing to the database: {e}")

@app.route('/history', methods=['POST'])
def get_history():
    with app.app_context():
        data = request.get_json()
        uid = data.get('uid', None)

        if uid is None:
            return make_response(jsonify({"error": "UID is required"}), 400)

        # 从数据库中检索历史记录
        records = QueryResponse.query.filter_by(uid=uid).all()
        # 如果没有找到任何记录，并且您希望在这种情况下添加一个新的UID记录
        if not records:
            # 由于这里没有实际的聊天，我们可能只是创建一个初始的空记录或特定的欢迎消息
            # new_record = QueryResponse(uid=uid, qid="initial", sub_qid="1", Query="", response="Welcome!")  # 可以根据需要自定义消息
            # db.session.add(new_record)
            # db.session.commit()
            # safe_commit()
            return jsonify([{"uid": uid, "qid": "", "sub_qid": "", "Query": "", "response": ""}])

            # # 为了一致性，将新记录添加到要返回的历史记录中
            # records = [new_record]
        records = QueryResponse.query.filter_by(uid=uid, sub_qid="1").all()
        history = [{"uid": uid, "qid": record.qid, "sub_qid": record.sub_qid, "Query": record.Query, "response": record.response} for record in records]
        # history = [{"qid": record.qid, "sub_qid": record.sub_qid, "Query": record.Query, "response": record.response} for record in records]
        return jsonify(history)

@app.route('/qid', methods=['POST'])
def get_records_by_qid():
    with app.app_context():
        data = request.get_json()
        uid = data.get('uid', None)
        qid = data.get('qid', None)

        # 检查是否提供了必要的数据
        if uid is None or qid is None:
            return make_response(jsonify({"error": "Both UID and QID are required"}), 400)

        # 从数据库中检索与给定uid和qid相关的所有记录
        records = QueryResponse.query.filter_by(uid=uid, qid=qid).order_by(QueryResponse.sub_qid).all()

        # 格式化结果
        formatted_records = [{"uid": record.uid, "qid": record.qid, "sub_qid": record.sub_qid, "Query": record.Query, "response": record.response} for record in records]

        return jsonify(formatted_records)

@app.route('/clear', methods=['POST'])
def clear_history():
    with app.app_context():
        data = request.get_json()
        uid = data.get('uid', None)
        qid = data.get('qid', None)

        # 检查是否提供了必要的数据
        if uid is None or qid is None:
            return make_response(jsonify({"error": "Both UID and QID are required"}), 400)

        # 从数据库中删除特定uid和qid的记录
        QueryResponse.query.filter_by(uid=uid, qid=qid).delete()
        safe_commit()

        # 返回结果
        return jsonify({"result": "History for UID {} and QID {} has been removed.".format(uid, qid)})


@app.route('/', methods=['POST'])
def chat():
    with app.app_context():
        data = request.get_json()
        uid = data.get('uid', None)
        qid = data.get('qid', None)
        Query = data.get('Query', None)

        if request_queue.full():
            return jsonify({"uid": uid, "qid": qid, "response": "busy"})
        # 查询数据库，如果存在，就增加sub_qid，如果不存在，就创建新条目
        last_record = QueryResponse.query.filter_by(uid=uid, qid=qid).order_by(QueryResponse.id.desc()).first()
        # last_record = QueryResponse.query.all()
        if last_record:
            new_sub_qid = str(int(last_record.sub_qid) + 1)  # 从最后一条记录中获取sub_qid，并将其值加一
            # new_record = QueryResponse(uid, qid, new_sub_qid, query, "等待响应")
        else:#这里对一开始的query也需要增加唯一的sub_qid
            new_sub_qid = "1"  # 这里对一开始的query也需要增加唯一的sub_qid
        
        new_record = QueryResponse(uid, qid, new_sub_qid, Query, "等待响应")

        db.session.add(new_record)
        # db.session.commit()
        safe_commit()

        # 当用户提交请求时，将其添加到特定的房间
        # join_room(user_sid)  # 使用用户的 UID 作为房间名
        request_queue.put((uid, qid, new_sub_qid, Query))
        return jsonify({"status": "received"}), 202  # 202 是 HTTP 的 "Accepted" 状态码

    

def process_queue():
    while True:
        uid, qid, sub_qid, Query = request_queue.get()
        # print("in process_queue")
        # print(type(uid))

        with app.app_context():
            # 检查关键字，并进行逻辑处理
            # if Query.strip().lower() == "clear":
            #     # 从数据库中删除特定uid和qid的记录
            #     QueryResponse.query.filter_by(uid=uid, qid=qid).delete()
            #     safe_commit()

                # 通知客户端历史记录已被清除
                # 检查我们是否知道该用户的sid
                # user_sid = connected_users_sid.get(uid)
                # if user_sid:
                #     socketio.emit('chat_response', {"uid": uid, "qid": qid, "sub_qid": sub_qid, "response": "History for UID {} has been removed.".format(uid)}, room=user_sid)
                # continue

            response = ""
            # 直接使用数据库中的记录来构建聊天历史。
            records = QueryResponse.query.filter_by(uid=uid, qid=qid).order_by(QueryResponse.sub_qid).all()
            chat_history = [(record.Query, record.response) for record in records if record.response != "等待响应"]

            # 配置chat_model的参数
            gen_kwargs = {
                "top_p": 0.4,
                "top_k": 0.3,
                "temperature": 0.95,
                "num_beams": 1,
                "max_length": 4096,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.2,
            }
            # for new_text in chat_model.stream_chat(query, user_history, input_kwargs=gen_kwargs):
            #     response += new_text
            for new_text in chat_model.stream_chat(Query, chat_history, input_kwargs=gen_kwargs):
                response += new_text

                # user_history.append((query, response))
            
            # 在数据库中查找或创建新记录
            latest_record = QueryResponse.query.filter_by(uid=uid, qid=qid, sub_qid=sub_qid).first()
            if latest_record is None:
                # 如果记录不存在，创建新的记录
                latest_record = QueryResponse(uid=uid, qid=qid, sub_qid=sub_qid, Query=Query, response=response)
                db.session.add(latest_record)
            else:
                # 如果记录存在，更新响应
                latest_record.response = response
            # 提交更改
            safe_commit()
            # socketio.emit('chat_response', {"qid": qid, "response": response}, room=uid, namespace='/')
            # 检查我们是否知道该用户的sid
            # for stored_uid, stored_sid in connected_users_sid.items():
            #     print(type(stored_uid))
            #     print(f"Stored UID: {stored_uid}, SID: {stored_sid}")
            with connected_users_sid_lock:
                # print(connected_users_sid)
                # print("before chat_response")
                user_sid = connected_users_sid.get(str(uid))
            socketio.emit('chat_response', {"uid": uid, "qid": qid, "sub_qid": sub_qid, "response": response}, room=user_sid)
            print(f"Sending chat_response to UID {uid} with SID {user_sid}")
            request_queue.task_done()

@socketio.on('connect', namespace='/')
def handle_connect():
    print('Client connected')

@socketio.on('send_uid', namespace='/')
def handle_send_uid(json):
    uid = json['uid']
    # print("in send_uid")
    # print(type(uid))
    user_sid = request.sid

    with connected_users_sid_lock:        
        connected_users_sid[str(uid)] = user_sid  # 存储客户端的 sid
    join_room(user_sid)  # 使用客户端的 sid作为room名称
    print(f'User {uid} joined room {user_sid}')

@socketio.on('disconnect', namespace='/')
def handle_disconnect():
    print('Client disconnected')
    user_sid = request.sid # 获取断开连接的客户端的 sid
    uid_to_remove = None

    # 查找具有相应 sid 的 uid
    for uid, sid in connected_users_sid.items():
        if sid == user_sid:
            uid_to_remove = uid
            break

    # 如果找到了，从字典中删除该用户
    with connected_users_sid_lock:
        if uid_to_remove:
            del connected_users_sid[uid_to_remove]
    print(f"User {uid_to_remove} with SID {user_sid} disconnected")


if __name__ == "__main__":
    with app.app_context():  # 创建应用上下文
        db.create_all()  # 在应用上下文内创建数据库表
        db.session.commit()  # 确保在相同的应用上下文中提交任何变更
    socketio.start_background_task(process_queue)
    socketio.run(app, host='0.0.0.0', port=5000)