def dec_to_bin(num, bits):
    if bits == 0:
        return
    dec_to_bin(num//2, bits-1)
    print(num%2, end="")

def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        N = int(input())
        if N<0 or N>255:
            exit()
        list.append(N)
    for num in list:
        dec_to_bin(num, 8)
        print()

if __name__ == "__main__":
    read()