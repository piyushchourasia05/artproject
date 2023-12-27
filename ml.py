import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import skimage.io
import skimage.transform
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.preprocessing.image import ImageDataGenerator
import os
artists = pd.read_csv('../Final Project/artists.csv')
artists.shape

artists.head() 

artists.dtypes
artists.isna().sum()
artists.groupby("nationality").sum().reset_index().sort_values("paintings",ascending=False)
artists.loc[(artists.nationality == "French,British")|(artists.nationality == "French,Jewish,Belarusian")|(artists.nationality == "German,Swiss")|(artists.nationality == "Spanish,Greek")
plt.figure(figsize=(18,12))
sns.barplot(orient = 'h',data=artists.groupby("nationality").sum().reset_index().sort_values("paintings",ascending=False),y="nationality",x="paintings")
plt.xlabel("Nationality")
plt.ylabel("Number of paintings")

plt.title("Most proficient nationalities over the top 50 artists",fontsize=15)

artists[["name","paintings"]].sort_values("paintings",ascending=False)
plt.figure(figsize=(18,12))
sns.barplot(data=artists[["name","paintings"]].sort_values("paintings",ascending=False),y="name",x="paintings",orient='h')
plt.ylabel("Names of painters")
plt.xlabel("Number of paintings")
plt.title("Ranking of painters from their number of paintings",fontsize=15)
plt.show()

pd.DataFrame(artists.genre.value_counts()).reset_index().sort_values("index")
plt.figure(figsize=(18,12))
sns.barplot(data=pd.DataFrame(artists.genre.value_counts()).reset_index(),y="index",x="genre",orient='h')
plt.ylabel("Genre")
plt.xlabel("Occurrence of genre")
plt.title("Genre representation over the top 50 artists",fontsize=15)
artists2=artists.copy()
dct_genre={"Abstract Expressionism":"Expressionism","Surrealism,Impressionism":"Surrealism","Early Renaissance":"Renaissance","Expressionism,Abstractionism" :"Expressionism","Expressionism,Abstractionism,Surrealism":"Expressionism","High Renaissance":"Renaissance","High Renaissance,Mannerism":"Renaissance","Impressionism,Post-Impressionism":"Impressionism","Northern Renaissance":"Renaissance","Primitivism,Surrealism":"Primitivism","Proto Renaissance":"Renaissance","Realism,Impressionism":"Realism","Social Realism,Muralism":"Realism"," Symbolism,Art Nouveau":"Symbolism","Symbolism,Expressionism":"Symbolism","Symbolism,Post-Impressionism":"Symbolism","Symbolism,Art Nouveau":"Symbolism"}
artists2.genre=artists2.genre.map(dct_genre).fillna(artists2.genre)
pd.DataFrame(artists2.genre.value_counts()).reset_index().sort_values("index")
artists2
plt.figure(figsize=(18,12))
sns.barplot(data=pd.DataFrame(artists2.genre.value_counts()).reset_index(),y="index",x="genre",orient='h')
plt.ylabel("Genre")
plt.xlabel("Occurrence of genre")

plt.title("Genre representation over the top 50 artists",fontsize=15)

artists2.groupby('genre').sum().sort_values("paintings",ascending=False).reset_index()
plt.figure(figsize=(18,8))
sns.barplot(data=artists2.groupby('genre').sum().sort_values("paintings",ascending=False).reset_index(),x="genre",y="paintings")
plt.xlabel("Genre")
plt.ylabel("Number of paintings")
plt.xticks(rotation='vertical')
plt.title("Ranking of number of paintings per genre",fontsize=15)
genre=list(artists2.genre.unique())
genre
artists2.loc[artists2.genre == "Renaissance"]



images_dir = '../Final Project/images/images'
images_dir_2 = '../Final Project/Trying'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists2['name'].str.replace(' ', '_').values
artists_top_genre = artists2['genre'].values


for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))

artists2.head()
n = 5
fig, axes = plt.subplots(1, n, figsize=(20,10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)
    image = plt.imread(random_image_file)
    random_genre=' '.join([str(elem) for elem in list(artists2[artists.name.str.match(random_artist.replace('_', ' '))].genre)])
    axes[i].imshow(image)
    axes[i].set_title("Artist: " + random_artist.replace('_', ' ')+"\n"+"Genre:"+random_genre)
    axes[i].axis('off')
plt.savefig('demo.png', transparent=True)
plt.show()



genre
dct_artist=dict(zip(artists_top_name.tolist(),list(artists2.genre)))

os.listdir()
os.chdir( '../Final Project/images_resized_100' )
os.listdir()
from skimage.transform import resize
import matplotlib.image as mpimg
for i in os.listdir():
    for j in os.listdir(i):
        load_img_rz = np.array(Image.open(r'../Final Project/images_resized_100/'+str(i)+"/"+str(j)).resize((100,100)))
        Image.fromarray(load_img_rz).save(r'../Final Project/images_resized_100/'+str(i)+'/'+str(j))
from PIL import Image

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32")
    return data      
  X_list1=[]
X_list2=[]
index1=[]
for i in os.listdir("../Final Project/images_resized_100/"):
    for j in os.listdir(i):
    
        result = load_image(r'C:/Users/piyush/Final Project/images_resized_100/'+(i)+'/'+(j))
        if result.shape != (100, 100,3):
            index1.append(j)
            X_list1.extend([result])
            
        else:
            X_list2.extend([result])

X=np.array(X_list2)
X.shape
dct_images={j:i for i in os.listdir() if '.' not in i for j in os.listdir(i) if j.endswith("jpg") if j not in index1 }
class_names=genre
dct_class={"Expressionism":0,"Realism":1,"Impressionism":2,"Surrealism":3,"Byzantine Art":4,"Post-Impressionism":5,"Symbolism":6,"Renaissance":7,"Suprematism":8,"Cubism":9,"Baroque":10,"Romanticism":11,"Primitivism":12,"Mannerism":13,"Neoplasticism":14,"Pop Art":15}
y_df=pd.DataFrame(list(dct_images.values()),columns=["Genre"])
y_df
y_df.Genre=y_df.Genre.map(dct_class)
y=np.array(list(y_df.Genre))
X_train, X_test, y_train, y_test=train_test_split(X,y)
X_train = X_train / 255.0

X_test = X_test / 255.0
y_train_df=pd.DataFrame(list(y_train),columns=["genre"])
y_train_df.genre.value_counts()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(16)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=10)


model.save('final_model_4.h5')

