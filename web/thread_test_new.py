# import socketio
# import threading

# sio = socketio.Client()
# sio.connect('http://localhost:5000/chat')

# @sio.on('chat_response')
# def on_response(data):
#     print(f"Received response for client_id {data['client_id']}: {data['response']}")

# def send_query(client_id):
#     sio.emit('chat', {'client_id': client_id, 'query': 'hello'})

# threads = []

# # 模拟并发数为15
# for i in range(1):
#     t = threading.Thread(target=send_query, args=(f"client_{i}",))
#     t.start()
#     threads.append(t)

# for t in threads:
#     t.join()

# sio.disconnect()


# import requests
# from concurrent.futures import ThreadPoolExecutor

# BASE_URL = 'http://127.0.0.1:5000/chat'
# QUERY = 'hello'
# NUM_REQUESTS = 1

# def send_request(i):
#     client_id = f"test_client_{i}"
#     response = requests.post(BASE_URL, json={"client_id": client_id, "query": QUERY})
#     print(response)
#     return response.json()

# def main():
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(send_request, i) for i in range(NUM_REQUESTS)]
#         for future in futures:
#             print(future.result())

# if __name__ == "__main__":
#     main()

#自己发送
# import requests
# import socketio

# sio = socketio.Client()
# client_id = "python_test_client"

# @sio.on('chat_response', namespace='/chat')
# def on_response(data):
#     if data['client_id'] == client_id:
#         print("[Response]:", data['response'])

# sio.connect('http://127.0.0.1:5000/chat')

# def send_message(message):
#     response = requests.post('http://127.0.0.1:5000/chat', json={"client_id": client_id, "query": message})
#     print(response.json())

# message = input("Enter message: ")
# while message.lower() != 'exit':
#     send_message(message)
#     message = input("Enter message: ")

# sio.disconnect()


# import requests
# import socketio
# import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # 定义一个SocketIO客户端
# #sio = socketio.Client()
# sio = socketio.Client(logger=True, engineio_logger=True)


# @sio.on('chat_response')
# def on_chat_response(data):
#     logger.info("Received response from server: %s", data)
#     print("Received response from server:", data)

# @sio.on('test_event')
# def on_test_event(data):
#     logger.info("Received test_event from server: %s", data)
#     print("Received test_event from server:", data)

# @sio.on('test_event_chat')
# def on_test_event_chat(data):
#     logger.info("Received test_event_chat from server: %s", data)
#     print("Received test_event_chat from server:", data)

# # 连接到SocketIO服务器
# logger.info("Connecting to server...")
# sio.connect('http://0.0.0.0:5000/')

# # 发送POST请求到服务器
# logger.info("Sending message to server...")
# response = requests.post('http://0.0.0.0:5000/', json={"client_id": "some_unique_client_id", "query": "hello"})
# print("POST request response:", response.json())

# # Keep the script running to listen for SocketIO responses
# try:
#     sio.wait()
# except KeyboardInterrupt:
#     sio.disconnect()

#一直连接
# import requests
# import socketio
# import logging
# import threading
# import time

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # 定义一个SocketIO客户端
# sio = socketio.Client(logger=True, engineio_logger=True)

# @sio.on('chat_response')
# def on_chat_response(data):
#     logger.info("Received response from server: %s", data)
#     print("Received response from server:", data)

# @sio.on('test_event')
# def on_test_event(data):
#     logger.info("Received test_event from server: %s", data)
#     print("Received test_event from server:", data)

# @sio.on('test_event_chat')
# def on_test_event_chat(data):
#     logger.info("Received test_event_chat from server: %s", data)
#     print("Received test_event_chat from server:", data)

# @sio.on('message')
# def on_message(data):
#     logger.info("Received default message from server: %s", data)
#     print("Received default message from server:", data)

# def socket_io_thread():
#     # 连接到SocketIO服务器
#     logger.info("Connecting to server...")
#     sio.connect('http://0.0.0.0:5000/')
    
#     # Keep the script running to listen for SocketIO responses
#     try:
#         sio.wait()
#     except KeyboardInterrupt:
#         sio.disconnect()

# # 创建一个线程来运行SocketIO连接
# socket_io_thread = threading.Thread(target=socket_io_thread)
# socket_io_thread.start()

# # Ensure the SocketIO connection is ready before sending POST request
# while not sio.connected:
#     time.sleep(0.1)

# # 发送POST请求到服务器
# logger.info("Sending message to server...")
# response = requests.post('http://0.0.0.0:5000/', json={"client_id": "some_unique_client_id", "query": "hello"})
# print("POST request response:", response.json())

# # 主线程中持续监听 chat_response 事件
# try:
#     while True:
#         time.sleep(1)  # Instead of a tight loop, just sleep for a second to conserve CPU usage
# except KeyboardInterrupt:
#     pass

# # Safely disconnect and cleanup
# sio.disconnect()
# socket_io_thread.join()

import requests
import socketio
import logging
import threading
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sio = socketio.Client(logger=True, engineio_logger=True)

# ... [Event handlers remain unchanged]
@sio.on('chat_response')
def on_chat_response(data):
    logger.info("Received response from server: %s", data)
    print("Received response from server:", data)

# @sio.on('test_event')
# def on_test_event(data):
#     logger.info("Received test_event from server: %s", data)
#     print("Received test_event from server:", data)

# @sio.on('test_event_chat')
# def on_test_event_chat(data):
#     logger.info("Received test_event_chat from server: %s", data)
#     print("Received test_event_chat from server:", data)

# @sio.on('message')
# def on_message(data):
#     logger.info("Received default message from server: %s", data)
#     print("Received default message from server:", data)

def socket_io_thread():
    logger.info("Connecting to server...")
    sio.connect('http://0.0.0.0:5000/')
    
    try:
        sio.wait()
    except KeyboardInterrupt:
        sio.disconnect()

socket_io_thread = threading.Thread(target=socket_io_thread)
socket_io_thread.start()

# Ensure the SocketIO connection is ready before sending POST requests
while not sio.connected:
    time.sleep(0.1)

# Send 15 POST requests with different client_id values
for i in range(1, 16):
    client_id = f"client_{i}"
    logger.info(f"Sending message to server with client_id: {client_id}...")
    response = requests.post('http://0.0.0.0:5000/', json={"client_id": client_id, "query": "hello"})
    print(f"POST request response for client_id {client_id}:", response.json())
    time.sleep(1)  # Added to avoid overloading the server in quick succession

# Main thread continues listening for chat_response events
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

# Safely disconnect and cleanup
sio.disconnect()
socket_io_thread.join()


