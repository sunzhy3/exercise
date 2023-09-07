def clear_list(l):
    # l = []
    l.append(4)
    l = []

ll = [1,2,3]
clear_list(ll)
print(ll)

def fl(l=[1]):
    l.append(1)
    print(l)
fl()
fl()