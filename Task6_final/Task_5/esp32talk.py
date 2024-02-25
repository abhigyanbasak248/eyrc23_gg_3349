import socket
import json

host = '172.20.10.2'  # Replace with the IP address of your ESP32
port = 80

data_to_send = [1, 2, 3, 4, 5]  # Replace with your list of numbers

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((host, port))
    json_data = json.dumps(data_to_send)
    s.sendall(json_data.encode())
    data = s.recv(1024)

print('Received:', data.decode())
