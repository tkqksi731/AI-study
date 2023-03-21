# 섞인 위치값에 따라서 데이터 내보내기
import numpy as np
a = np.array([10,20,30,40,50])
print(a)

b = a[np.array([1,0,2])]

indicies = np.arange(len(a))
np.random.shuffle(indicies)
print(indicies)
b = a[indicies]
print(b)

