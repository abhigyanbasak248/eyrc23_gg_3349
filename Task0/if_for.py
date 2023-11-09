def perform(num):
    if num == 0:
        num +=3
    elif num%2 == 0:
        num *= 2
    else:
        num = num ** 2
    print(num, end=" ")


def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        n = int(input())
        if n<0 or n>100:
            exit()
        list.append(n)
    for num in list:
        for i in range(num):
            perform(i)
        print()

if __name__ == "__main__":
    read()