#티어레이터 
#이터레이터란? 클래스 내부에 실제 데이터의 저장형식과 관계없이.
#             클래스 외부(객체)에서 동일한 방식으로 내부의 데이터를
#             차례대로 접근할 수 있도록 약속한 인터페이스
#             파이썬내의 컬렉션 클래스들(list, tuple, set, dict등)
#             이 실제 내부에 어떤 형태로 데이터를 저장했는지 몰라도
#             외부 사용자는 동일한 접근 방법을 통해서 데이터를 접근할 수 있다,
#             이를 이터레이터라고 한다
a = [1,2,3,4,5]
t = iter(a) #이터레이터 객체를 만들어서 반환, 데이터 처음부터접근

print(next(t)) # 다음 데이터로, 현재 내가 있는 위치가 저장
print(next(t))
print(next(t))
print(next(t))
print(next(t))

mycolor = {"red":"빨간색", "green":"초록색", "blue":"파란색"}
it = iter(mycolor)
print(next(it))
print(next(it))
print(next(it))

for item in a:
    print(item)
print("-"*20)

# 직접 이터레이터를 만들어 보자
def getNumber(limit):
    for i in range(1, limit+1):
        yield i #리턴 - 함수를 종료
                #yield - 값 하나보내고 함수가 종료 안됨
                #값을 보내고 남아 있ㅇ므

b = iter(getNumber(5))
print( next(b))
print( next(b))
print( next(b))
print( next(b))

def getChar(s):
    for char in s:
        yield char

c = iter(getChar("제너레이터"))
print(next(c))
print("-"*20)
for c in getChar("머신러닝과딥러닝"):
    print(c)

