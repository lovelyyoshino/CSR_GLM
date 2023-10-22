// import socketio from 'socket.io-client';
// import axios from 'axios';

// const socketio = require('socket.io-client');
// const axios = require('axios');

// const sio = socketio('http://127.0.0.1:5000');

// sio.on('connect', () => {
//     console.log('Client connected');
// });

// sio.on('disconnect', () => {
//     console.log('Client disconnected');
// });

// sio.on('chat_response', (data) => {
//     console.log('Received response from server:', data);
// });

// // Ensure the SocketIO connection is ready before sending POST requests
// setTimeout(() => {
//     for (let i = 1; i <= 15; i++) {
//         const client_id = `client_${i}`;
//         console.log(`Sending message to server with client_id: ${client_id}...`);
//         axios.post('http://127.0.0.1:5000/', {
//             client_id: client_id,
//             query: 'hello'
//         })
//         .then((response) => {
//             console.log(`POST request response for client_id ${client_id}:`, response.data);
//         })
//         .catch((error) => {
//             console.error(`Error in POST request for client_id ${client_id}:`, error);
//         });
//     }
// }, 1000);

const socketio = require('socket.io-client');
const axios = require('axios');

// 使用你的服务器的IP地址和端口号
const sio = socketio('http://117.174.101.198:5000');

sio.on('connect', () => {
    console.log('Client connected');
});

sio.on('disconnect', () => {
    console.log('Client disconnected');
});

sio.on('chat_response', (data) => {
    console.log('Received response from server:', data);
});

// 确保SocketIO连接准备好后再发送POST请求
setTimeout(() => {
    for (let i = 1; i <= 15; i++) {
        const client_id = `client_${i}`;
        console.log(`Sending message to server with client_id: ${client_id}...`);
        axios.post('http://117.174.101.198:5000/', {
            client_id: client_id,
            query: 'hello'
        })
        .then((response) => {
            console.log(`POST request response for client_id ${client_id}:`, response.data);
        })
        .catch((error) => {
            console.error(`Error in POST request for client_id ${client_id}:`, error);
        });
    }
}, 1000);

