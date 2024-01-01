import pandas as pd
import preprocesado
import predict
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import nltk 
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from textblob import TextBlob

body_part = wn.synsets('body_part')[0]
injury_synset = wn.synsets('injury', lang='eng')[0]
stopwords = set(stopwords.words("english"))
porter=nltk.PorterStemmer()

def cargar_train():
    df = pd.read_csv("datasets/train.csv")
    columnas_necesarias = ['ClaimDescription', 'UltimateIncurredClaimCost']
    todas_las_columnas = df.columns.tolist()
    columnas_a_eliminar = [col for col in todas_las_columnas if col not in columnas_necesarias]
    df = df.drop(columns=columnas_a_eliminar)
    return df


def CorectText(df):
    for i in range(0, len(df)):
        print(i)
        new_doc = TextBlob(df.at[i, "ClaimDescription"])
        result = new_doc.correct()
        if result.lower() != df.at[i, "ClaimDescription"].lower():
            print(result + " :: " + df.at[i, "ClaimDescription"])
            df.at[i, "ClaimDescription"] = result
    return df        

def Valores_Nulos(df):
    mas_frecuente = df['Claim_Body_Parts'].mode()[0]
    df['Claim_Body_Parts'].fillna(mas_frecuente, inplace=True)
    mas_frecuente = df['Claim_Injuries'].mode()[0]
    df['Claim_Injuries'].fillna(mas_frecuente, inplace=True)
    return df



def Codificar(df):
    label_encoder = LabelEncoder()
    dataframe = df.copy()
    dataframe['Claim_Body_Parts'] = label_encoder.fit_transform(dataframe['Claim_Body_Parts'])
    dataframe['Claim_Injuries'] = label_encoder.fit_transform(dataframe['Claim_Injuries'])
    return dataframe



def main():
    df = cargar_train()
    #df = preprocesado.preprocesado2(df)
    df = preprocesado.Preprocesado(df)
    #df = Codificar(df)
    df = Valores_Nulos(df)
    df.to_csv("Task 2/preprocesing.csv", sep=",", index=False)
    #predict.crearmodelo_NN_tfidf(df)
    #predict.crearModelo_RandomForest(df)
    predict.crearmodelo_NN(df)


if __name__ == "__main__":
    main()