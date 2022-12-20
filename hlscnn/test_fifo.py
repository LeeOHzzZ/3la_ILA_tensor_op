import os
import errno

SERVER_FIFO = "/home/yl29/tmp/hlscnn_ilator_server"
CLIENT_FIFO = "/home/yl29/tmp/hlscnn_ilator_client"

# make a client fifo
try:
    os.mkfifo(CLIENT_FIFO)
except OSError as oe:
    if oe.errno != errno.EEXIST:
        raise

cntr = 0

while True:
    print("Opening Server FIFO...")
    # the open command will be blocked if the other end of the pipe is not opened!
    # this is different from the C++ implementation
    server_fifo = open(SERVER_FIFO, "w")
    print("Server FIFO opened!")

    if cntr > 10:
        break
    print("starting simulation!")
    server_fifo.write("start!")
    # need this close command to send the information out
    server_fifo.close()
    cntr += 1

    print("Opening Client FIFO...")
    client_fifo = open(CLIENT_FIFO, "r")
    print("Client FIFO opened!")

    data = client_fifo.read()
    assert "finished" in data, data
    print(data)

server_fifo = open(SERVER_FIFO, "w")
server_fifo.write("stop")
server_fifo.close()