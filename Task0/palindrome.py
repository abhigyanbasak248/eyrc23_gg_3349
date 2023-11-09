def check(word):
    word = word.lower()
    word_rev = word [::-1]
    if word_rev!=word:
        print("It is not a palindrome")
    else:
        print("It is a palindrome")

def read():
    T = int(input())
    list = []
    if 1>T or T> 25:
        print("Invalid input")
        exit()
    for i in range(T):
        word = input()
        if len(word)>70:
            print("Invalid input")
            exit()
        list.append(word)
        
    for word in list:
        check(word)

if __name__ == "__main__":
    read()