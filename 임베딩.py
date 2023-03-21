from  keras.datasets import imdb
from keras import preprocessing

max_features = 10000 #최대특성개수, 특성으로 사용할 단어의 수
maxlen = 20 #사용할 텍스트의 길이

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = max_features
)
#이미 시퀀스화 되어 있는 데이터이다.
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# list -> samples, maxlen 크기의 2D 정수텐서로 변환
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add( Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())

model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop",
                loss="binary_crossentropy",
                metrics=["acc"])

history = model.fit( x_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.2)

results = model.evaluate(x_test, y_test)
print(results)

pred = model.predict(x_test)
pred_class = model.predict_classes(x_test)
print(pred_class[:40])
print(y_test[:40])