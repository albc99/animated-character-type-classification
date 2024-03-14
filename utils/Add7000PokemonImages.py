import pypokedex
import os
from sklearn.model_selection import train_test_split
import shutil

directory = '../data/PokemonData/'

pokemon = pypokedex.get(name="sandslash")
print(pokemon.types[0])

files = []
types = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if not os.path.isfile(f):
        slash_index = f.rfind('/') + 1
        pokemon_name = f[slash_index:]
        try:
            pokemon = pypokedex.get(name=pokemon_name)
            pokemon_type = pokemon.types[0].title()
            subdirectory = f[:slash_index] + pokemon_name + '/'
            for subfilename in os.listdir(subdirectory):
                f_sub = os.path.join(subdirectory, subfilename)
                if os.path.isfile(f_sub):
                    files.append(f_sub)
                    types.append(pokemon_type)
        except pypokedex.exceptions.PyPokedexHTTPError:
            continue

X_train, X_test, y_train, y_test = train_test_split(files, types, test_size = 0.2, stratify=types, random_state=42)
print(X_train, X_test, y_train, y_test)
# move training images
for i in range(len(X_train)):
    shutil.move(X_train[i], f'../data/images/train/{y_train[i]}')

# move testing images
for i in range(len(X_test)):
    shutil.move(X_test[i], f'../data/images/test/{y_test[i]}')

