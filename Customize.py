import pyspark.sql.functions as F
import pandas as pd

JAVA_PATH = ''
CPRD_CODE_PATH = '' # CPRD code path
COHORT_SAVE_PATH = '' # save cohort
CBEHRT_PATH = ''
MODEL_SAVE_PATH = '' # save model
code_path = ''
sypmtom_path = ''

def get_medcode(med_read_term, disease='PD'):
    if disease == 'PD':
        return []
    
    elif disease == 'AD':
        AD_readcode = pd.read_csv(code_path + 'AD.csv')['Code'].astype('string').tolist()
        AD_medcode = med_read_term.where(F.col("readcode").isin(AD_readcode)).select("medcode").toPandas()['medcode'].tolist()
        return AD_medcode
    
    elif disease == 'PARM':
        PD_snow_code = pd.read_csv(code_path + 'PD.csv')['Code'].astype('string').tolist()
        PD_medcode = med_read_term.where(F.col("Snomedcode").isin(PD_snow_code)).select("medcode").toPandas()['medcode'].tolist()
        return PD_medcode

def get_icdcode(disease='PD'):
    if disease == 'PD':
        code = ['G20', 'G200']
        # code = ['F023', 'G20', 'G210', 'G211', 'G212', 'G213', "G214", 'G218', 'G219', 'G22', "G232"]

    elif disease == 'AD':
        code = ['F00', 'F000', 'F001', 'F002', 'F009', 'G30', 'G300','G301', 'G308','G309']
        # code1 = ["F00"+ str(i) for i in range(10)]
        # # code2 = ["F02"+ str(i) for i in range(10)]
  
    elif disease == 'PARM':
        code = ['G20', 'G21', 'G210','G211', 'G212', 'G213', 'G214', 'G218', 'G219', 'G22', 'G230', 'G231', 'G232', 'G238', 'G239', 'G259', 'G26', 'G903']
    return code


def get_symptom_medcode(med_read_term, symptom):
    if symptom == 'fall':
        read_code = pd.read_csv(sypmtom_path + 'red/fall.csv')['Read code'].astype('string').tolist()
        code = med_read_term.where(F.col("readcode").isin(read_code)).select("medcode").toPandas()['medcode'].tolist()
 
    elif symptom == 'language':
        read_code = pd.read_csv(sypmtom_path + 'red/language.csv')['Read code'].astype('string').tolist()
        code = med_read_term.where(F.col("readcode").isin(read_code)).select("medcode").toPandas()['medcode'].tolist()
    
    elif symptom == 'memory':
        read_code = pd.read_csv(sypmtom_path + 'red/memory.csv')['Read code'].astype('string').tolist()
        code = med_read_term.where(F.col("readcode").isin(read_code)).select("medcode").toPandas()['medcode'].tolist()
    
    elif symptom == 'confusion':
        read_code = pd.read_csv(sypmtom_path + 'red/confusion.csv')['Read code'].astype('string').tolist()
        code = med_read_term.where(F.col("readcode").isin(read_code)).select("medcode").toPandas()['medcode'].tolist()
    elif symptom == 'sleep':
        read_code = pd.read_csv(sypmtom_path + 'red/sleep.csv')['Read code'].astype('string').tolist()
        code = med_read_term.where(F.col("readcode").isin(read_code)).select("medcode").toPandas()['medcode'].tolist()

    elif symptom == 'dementia':
        sno_code = pd.read_csv(sypmtom_path + 'med/Dementia.csv')['Code'].astype('string').tolist()
        code = med_read_term.where(F.col("snocode").isin(sno_code)).select("medcode").toPandas()['medcode'].tolist()

    elif symptom == 'depression':
        sno_code = pd.read_csv(sypmtom_path + 'med/Depression.csv')['Code'].astype('string').tolist()
        code = med_read_term.where(F.col("snocode").isin(sno_code)).select("medcode").toPandas()['medcode'].tolist()

    elif symptom == 'constipation':
        sno_code = pd.read_csv(sypmtom_path + 'med/constipation.csv')['Code'].astype('string').tolist()
        code = med_read_term.where(F.col("snocode").isin(sno_code)).select("medcode").toPandas()['medcode'].tolist()
    
    elif symptom == 'mci':
        sno_code = pd.read_csv(sypmtom_path + 'med/mci.csv')['Code'].astype('string').tolist()
        code = med_read_term.where(F.col("snocode").isin(sno_code)).select("medcode").toPandas()['medcode'].tolist()
        
    return code