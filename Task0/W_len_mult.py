def count(words):
    c = 0
    for i in range(len(words)-1):
        c += 1
        print(len(words[i]), end = ",")
    print(len(words[c]), end = "")

def read():
    T = int(input())
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    list = []
    for i in range(T):
        N = input()
        N = N[1:]
        if not N.isalpha():
            exit()
        list.append(N)
    for sent in list:
        words = sent.split()
        count(words)
        print()

if __name__ == "__main__":
    read()