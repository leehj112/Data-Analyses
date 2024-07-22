# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:48:20 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:36:19 2023

@author: Solero
"""

# 소켓(Socket) 프로그래밍
# 서버 프로그램

from socket import *

# host = "127.0.0.1"
host = "192.168.71.200"
port  = 9999

server_socket  = socket(AF_INET, SOCK_STREAM)
server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

server_socket.bind((host, port))

print("Listening...")

server_socket.listen()

client_socket, addr = server_socket.accept()

print("Connected by", addr)

while True:
    data = client_socket.recv(1024)
    if not data:
        break
    
    print("Received from:", addr, data.decode())
    
    data += ", 잘 받음!!!".encode()
    client_socket.sendall(data)
    
client_socket.close()
server_socket.close()

    
        