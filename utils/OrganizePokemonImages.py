import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import os

pokedex = pd.read_csv('../data/Pokedex_Cleaned.csv', encoding='latin-1')

not_types = {"Null", "Male", "Female", "Rockruff"}
pokedex = pokedex.loc[~pokedex['Primary Type'].isin(not_types)]  # remove rows with invalid types
primary_types = pokedex['Primary Type']
primary_types.value_counts()  # bad rows are gone

# Drop irrelevant rows, columns, reorder name
pokemon_names = pokedex['Name'].copy()
pokedex_relevant = pokedex.drop(['#', 'Name', 'Secondary Type', 'Total', 'Variant'], axis=1)
pokedex_relevant['Name'] = pokemon_names
pokedex_relevant = pokedex_relevant.drop_duplicates(subset=["Name"], keep='last')

# extract features and labels
features = pokedex_relevant.iloc[:, -1].values
labels = pokedex_relevant.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, stratify=labels, random_state=42)
train_name_to_type = {name.lower():value for name,value in tuple(zip(X_train, y_train))}
print(train_name_to_type, len(train_name_to_type))
test_name_to_type = {name.lower():value for name,value in tuple(zip(X_test, y_test))}
print(test_name_to_type, len(test_name_to_type))
path = '../data/images'
files = os.listdir(path)

not_there = []
for i in tqdm(files):
    filename = path + '/'+i
    if ".png" in filename:
        slash_index = filename.rfind('/')+1
        dot_index = filename.rfind('.')
        pokemon_name = filename[slash_index:dot_index]
        print(filename)
        if pokemon_name in train_name_to_type:
            shutil.move(filename, f'../data/images/train/{train_name_to_type[pokemon_name]}')
        elif pokemon_name in test_name_to_type:
            shutil.move(filename, f'../data/images/test/{test_name_to_type[pokemon_name]}')
        else:
            not_there.append(pokemon_name)

print("Missing from dataset:", len(not_there), not_there)

