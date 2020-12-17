#%% import libraries
#Modeling
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Dense,Embedding,Dropout

from keras.preprocessing.sequence import pad_sequences

#%% Modeling 
class Modeling:

    def __init__(self, df):
        self.df = df

    def LSTM(self, X_train_pad,Y_train_f, X_val_pad,Y_val_f):
        
        #Creating a Model
        self.model=Sequential()
        # Embedded layer that uses 64 length vectors to represent each word
        self.model.add(Embedding(199973, 64, input_length = 48506))
        self.model.add(Dropout(0.6))
        self.model.add(Bidirectional(LSTM(80, return_sequences=True))) 
        self.model.add(Bidirectional(LSTM(160)))
        self.model.add(Dense(7,activation='softmax'))
        print(self.model.summary())

        #Compiling and running the model
        #The efficient ADAM optimization algorithm is used
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        history = self.model.fit(X_train_pad,Y_train_f,epochs=5,validation_data=(X_val_pad,Y_val_f))

        return history

    @classmethod
    def get_key(self, value):
        dictionary={'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5, 'normal':6}
        for key,val in dictionary.items():
            if (val==value):
                return key
         


    def predict(self, tokenizer, email):
        obj = Modeling
        sentence_lst=[]
        sentence_lst.append(email)
        sentence_seq=tokenizer.texts_to_sequences(sentence_lst)
        sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
        ans = obj.get_key(self.model.predict_classes(sentence_padded))
        print("The emotion predicted is",ans)


# %%
