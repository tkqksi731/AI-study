import numpy as np

#스칼라 텐서
a = np.array(12)
print(type(a), a.shape)

#벡터 텐서 - 파이썬의 list -> numpy의 벡터로 전환
# [] -> 벡터, [[ ]] -> 행렬, [[[ ]]] -> array 또는 3d텐서
# [[[[ ]]]] -> 4d tensor, [[[[[ ]]]]] -> 5d 텐서 구조
b = np.array( [1,2,3,4,5] )
print(type(b), b.shape)
print(b)


print("------2d-------")
c = np.array ([[1,2,3],
                [4,5,6]])
print(type(c), c.shape)
print(c)

print("------3d-------")
d = np.array([[[1,2],[3,4]],
        [[5,6],[7,8]],
        [[9,10],[11,12]]])
print(type(d), d.shape)
print(d)

#벡터에 연산을 수행하면 벡터를 반환
y = 3*b + 4
print(y)

y = 3*c + 4
print(y)

#브로드캐스팅이 된다면 (1,"A"), (2,"B"), (3,"A"), (4,"b")
#zip은 현재 브로드캐스팅 안해줌
a = [1,2,3,4]
b = ["A", "B"]
for item in zip(a,b):
    print(item)

a = np.array([1,2,3,4,5,6,7,8,9]) # 벡터
b = a.reshape((3,3)) # 행렬
print( type(a), a.shape)
print( type(b), b.shape)

c = b.reshape(9,1)
print(type(c), c.shape)