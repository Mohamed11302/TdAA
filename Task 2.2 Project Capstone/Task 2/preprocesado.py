import pandas as pd
import unidecode
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk 
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import time

body_part = wn.synsets('body_part')[0]
injury_synset = wn.synsets('injury', lang='eng')[0]
stopwords = set(stopwords.words("english"))
porter=nltk.PorterStemmer()
symptom =  wn.synsets('symptom', lang='eng')[0]
mishap =  wn.synsets('mishap', lang='eng')[1]
synsets = [injury_synset, symptom, mishap]


synsets2 = ["cut", "strike", "strain", "fell", "burn", "fracture", "lacerate", "injury", "broken", "stress", "bruise", "laceration"]
def convert_tolower(df):
    for indice, fila in df.iterrows():
        cadena_original = fila["ClaimDescription"]
        df.at[indice, "ClaimDescription"] = cadena_original.lower()
    return df

def delete_repeated_words(df):
    df_nuevo = df.copy()

    # Itera sobre cada fila del dataframe
    for indice, fila in df_nuevo.iterrows():
        # Obtiene la cadena de texto de la columna especificada
        cadena_original = fila["ClaimDescription"]
        # Divide la cadena en palabras
        palabras = cadena_original.split()
        # Elimina las palabras duplicadas manteniendo el orden original
        palabras_sin_repetir = list(dict.fromkeys(palabras))
        # Une las palabras sin repetir de nuevo en una cadena
        nueva_cadena = ' '.join(palabras_sin_repetir)
        nueva_cadena = word_tokenize(unidecode.unidecode(nueva_cadena))
        cadena_2 =  [w for w in nueva_cadena if not w in stopwords]
        nueva_cadena2 = ' '.join(cadena_2)
        df_nuevo.at[indice, "ClaimDescription"] = nueva_cadena2
    df_nuevo.to_csv("pruebas_csv/delete_repeated_words.csv", sep=";", index=False)
    return df_nuevo

def combine_strings(array):
    string = ""
    for st in array:
        if st !="":
            string += " " + st
    if string == "":
        string = None
    else:
        string = string[1:]
    return string


def extract_body_parts(df):
    print("Extracting body_parts")
    partes_del_cuerpo = []
    claim_body_parts = []
    i = 0
    for description in df['ClaimDescription']:
        for token in word_tokenize(description):    
            
            #token_stem = porter.stem(token)
            token_stem = token
            i = i+1
            if is_body_part(token_stem):
                partes_del_cuerpo.append(token_stem)
            elif is_body_part_recursive(token_stem, 3):
                partes_del_cuerpo.append(token_stem)
        
        body_parts = combine_strings(partes_del_cuerpo[0:i])
        claim_body_parts.append(body_parts)
        partes_del_cuerpo = []
        i = 0
    df["Claim_Body_Parts"] = claim_body_parts
    df.to_csv("pruebas_csv/extracted_parts.csv", sep=";", index=False)
    df2 = df[df["Claim_Body_Parts"].isnull()]
    df2.to_csv("pruebas_csv/extracted_parts_nulls.csv", sep=";", index=False)
    return df


def try_injury(candidate):
    for ss in wn.synsets(candidate):
        name = ss.name().split(".", 1)[0]
        if name != candidate:
            continue
        for type in synsets:
            hit = type.lowest_common_hypernyms(ss)
            if hit and hit[0] == type:
                return True
    return False

def try_injury2(candidate):
    for ss in wn.synsets(candidate):
        name = ss.name().split(".", 1)[0]
        if name in synsets2:
            return True
    return False

def is_injury(token):
    if try_injury2(token):
        return True
    else:
        return False

def extract_injuries(df):
    print("Extracting injuries")
    injuries = []
    claim_injuries = []
    i = 0
    for description in df['ClaimDescription']:
        for token in word_tokenize(description):    
            
            token_stem = porter.stem(token)
            i = i+1
            if is_injury(token_stem) or is_injury(token):
                injuries.append(token_stem)
            #elif is_body_part_recursive(token_stem, 3):
             #   partes_del_cuerpo.append(token_stem)
        
        inj = combine_strings(injuries[0:i])
        claim_injuries.append(inj)
        injuries = []
        i = 0
    df["Claim_Injuries"] = claim_injuries
    df.to_csv("pruebas_csv/injuries.csv", sep=";", index=False)
    df2 = df[df["Claim_Injuries"].isnull()]
    df2.to_csv("pruebas_csv/injuries_nulls.csv", sep=";", index=False)
    return df


def count_synsets(df):
    print("Counting synsets")
    # Frecuencia de synsets
    synset_frequency = FreqDist()

    for description in df['ClaimDescription']:
        # Contar la frecuencia de cada synset en la lista de lesiones
        for token in word_tokenize(description):
            synsets = wn.synsets(token)
            for synset in synsets:
                name_parts = synset.name().split(".")
                if name_parts:
                    name = name_parts[0]
                    synset_frequency[name] += 1

    # Imprimir los 30 synsets con más apariciones
    print("Top 100 Synsets:")
    for synset, count in synset_frequency.most_common(100):
        print(f"{synset}: {count}")

    return df

def is_body_part(candidate):
    for ss in wn.synsets(candidate):
        name = ss.name().split(".", 1)[0]
        if name != candidate:
            continue
        hit = body_part.lowest_common_hypernyms(ss)
        if hit and hit[0] == body_part:
            return True
    return False




def obtener_holonomos(palabra):
    synsets = wn.synsets(palabra)
    if not synsets:
        return None
    holonomos = []
    for ss in synsets:
        # Obtiene los holónimos y agrega los nombres a la lista
        holonomos.extend([holonimo.name().split('.')[0] for holonimo in ss.part_holonyms()])

    return holonomos

def is_body_part_recursive(palabra, n):
    if n == 0:
        return False

    holonomos = obtener_holonomos(palabra)
    body_synset = wn.synsets('body')[0]
    if holonomos == None:
        return False
    else:
        for holonimo in holonomos:
            #print(f"Buscar holonimo ({n} veces): {holonimo}")
            if body_synset in wn.synsets(holonimo):
                return True

            # Llamada recursiva con n-1
            if is_body_part_recursive(holonimo, n - 1):
                return True

    return False









def Preprocesado(df):
    df = convert_tolower(df)
    df = delete_repeated_words(df)
    df = extract_body_parts(df)
    #df = count_synsets(df)
    df = extract_injuries(df)
    print(df.isnull().sum()) 
    return df



def delete_repeated_words2(df):
    df_nuevo = df.copy()

    for indice, fila in df_nuevo.iterrows():
        cadena_original = fila["ClaimDescription"]
        palabras = cadena_original.split()
        palabras = [porter.stem(palabra) for palabra in palabras]
        palabras_sin_repetir = list(dict.fromkeys(palabras))
        nueva_cadena = ' '.join(palabras_sin_repetir)

        nueva_cadena = word_tokenize(unidecode.unidecode(nueva_cadena))
        cadena_2 =  [w for w in nueva_cadena if not w in stopwords]
        nueva_cadena2 = ' '.join(cadena_2)
        df_nuevo.at[indice, "ClaimDescription"] = nueva_cadena2
    df_nuevo.to_csv("pruebas_csv/preprocesado2.csv", sep=";", index=False)
    return df_nuevo

def preprocesado2(df):
    df = convert_tolower(df)
    df = delete_repeated_words2(df)
    return df


