from Customize import JAVA_PATH, CPRD_CODE_PATH, COHORT_SAVE_PATH, get_medcode, get_icdcode

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['JAVA_HOME'] = JAVA_PATH

import sys 
sys.path.insert(0, CPRD_CODE_PATH)

import pyspark
import shutil
from utils.yaml_act import yaml_load
from CPRD.config.spark import spark_init, read_parquet
import pyspark.sql.functions as F
from pyspark.sql import Window
from CPRD.functions import tables, modalities
from CPRD.config.spark import read_txt, read_parquet
from CPRD.functions import merge
from utils.utils import save_obj, load_obj, create_folder
from CPRD.functions.cohort_select import Cohort
from CPRD.base.table import Diagnosis
import pandas as pd
import numpy as np
import itertools
from argparse import ArgumentParser
import logging

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args_ = dotdict({'params': os.path.join(CPRD_CODE_PATH, 'config','config.yaml')})
params = yaml_load(args_.params)
spark_params = params['pyspark']
spark = spark_init(spark_params)
file = params['file_path']
data_params = params['params']
pheno_dict = load_obj(file['PhenoMaps'])



parser = ArgumentParser()
parser.add_argument("--disease", type=str, default='AD')
parser.add_argument("--cohort_dir", type=str, default='AD_data')
parser.add_argument("--least_age", type=int, default=40)
parser.add_argument("--greatest_age", type=int, default=150)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--gp_code", type=str, default='snomed', choices=['read', 'snomed'])


args = parser.parse_args()
save_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir)
create_folder(save_path)


# add log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler( os.path.join(save_path, f'{args.disease}_cohort.log') )
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info("----------------------------------------------------")
logger.info("Beginning cohort generation with log written.")


diagnoses_path = ''
# process diagnoses codes
diagnoses = read_parquet(spark.sqlContext, diagnoses_path)
diag_cprd = diagnoses.filter(F.col('source')=='CPRD').select(['patid', 'eventdate', 'medcode']).withColumnRenamed('medcode', 'code')
diag_hes = diagnoses.filter(F.col('source')=='HES').select(['patid', 'eventdate', 'ICD']).withColumnRenamed('ICD', 'code')
diagnoses = diag_cprd.union(diag_hes)
diagnoses.show(5)



# CTS select medcode by two ways
med_read_term = read_txt(spark.sc, spark.sqlContext, file['med2read']).withColumnRenamed('MedCodeId', 'medcode').withColumnRenamed('CleansedReadCode', 'readcode') \
        .withColumn('medcode', F.col('medcode').cast('string')).select(['medcode', 'readcode','term'])
med_snow_term = read_txt(spark.sc, spark.sqlContext, file['med2read']).withColumnRenamed('SnomedCTConceptId', 'Snomedcode').withColumnRenamed('MedCodeId', 'medcode') \
        .withColumn('medcode', F.col('medcode').cast('string')).select(['medcode', 'Snomedcode','term'])

if args.gp_code == 'read':
    disease_medcode = get_medcode(med_read_term, args.disease)
elif args.gp_code == 'snomed':
    disease_medcode = get_medcode(med_snow_term, args.disease)

disease_icdcode = get_icdcode(args.disease)

assert len(disease_icdcode) > 0

CodeDict = {
'medcode': list(set( disease_medcode )), # GP code
'ICD10': list(set( disease_icdcode )), # Hes code
'prodcode': [], # medication-defined
'OPCS': []}

logger.info(f"The medcode of {args.disease} is {CodeDict['medcode']}")
logger.info(f"The icdcode of {args.disease} is {CodeDict['ICD10']}")

all_codes = list(itertools.chain.from_iterable(CodeDict.values()))
disease = diagnoses.filter(F.col('code').isin(*all_codes))

w = Window.partitionBy('patid').orderBy('eventdate')
disease = disease.withColumn('code', F.first('code').over(w)).groupBy('patid').agg(
    F.min('eventdate').alias('eventdate'),
    F.first('code').alias('code')
)

# extract the whole cohort
cohortselector = Cohort(least_year_register_gp=1, least_age=args.least_age, greatest_age=args.greatest_age)
demographics = cohortselector.standard_prepare(file, spark)

demographics = demographics.join(disease, 'patid', 'left').cache()

patient = tables.retrieve_patient(dir=file['patient'], spark=spark).select('patid','pracid')
demographics = demographics.join(patient, 'patid', 'inner')
logger.info(f"Total number of eligible patients linked with GP and HES is {demographics.count()}")

# extract the first diagnosis date 
earliest_diagnosis = diagnoses.groupBy('patid').agg(F.min('eventdate').alias('first_diagnosis_date'))
demographics = demographics.join(earliest_diagnosis, on='patid', how='left')

imd = modalities.retrieve_imd(file, spark)
demographics = demographics.join(imd, 'patid', 'inner')
logger.info(f"Number of patients with IMD is {demographics.count()}")



demographics = demographics.filter(F.col('eventdate').isNotNull())
logger.info(f'all {args.disease} patients is {demographics.count()}')
demographics = demographics.filter((demographics.eventdate >= '2005-01-01') & (demographics.eventdate <= '2018-01-01'))
logger.info(f'with incident {args.disease} between 2005-01-01 and 2018-01-01 is {demographics.count()}')
demographics = demographics.filter(demographics.eventdate < demographics.enddate)
logger.info(f'did not lost to followup before the incident {args.disease} or die with {args.disease} is {demographics.count()}')
demographics = demographics.filter(demographics.eventdate > F.col('least_gp_register_date'))
logger.info(f'at least 1 year GP registration before incident {args.disease} is {demographics.count()}')

demographics = demographics.filter(demographics.eventdate >= F.col(str(args.least_age)+'_dob'))
logger.info(f'at least {args.least_age} at the time of incident {args.disease} is {demographics.count()}')

if not os.path.exists(os.path.join(save_path, 'Cohort_CPRD')):
    demographics.write.parquet(os.path.join(save_path, 'Cohort_CPRD'))

logger.info("Cohort generation finished.")

logger.info("----------------------------------------------------")

# data split for validation
patient = tables.retrieve_patient(dir=file['patient'], spark=spark).select('patid','pracid')

unique_pracid_df = patient.select("pracid").distinct()

# Split the distinct pracid values into two datasets: 80% and 20%
internal_df, external_df = unique_pracid_df.randomSplit([0.8, 0.2], seed=args.seed)

logger.info(f"Total unique pracid values: {unique_pracid_df.count()}")
logger.info(f"Internal pracid counts (80%): {internal_df.count()}")
logger.info(f"External pracid counts (20%): {external_df.count()}")
logger.info("----------------------------------------------------")

if not os.path.exists(os.path.join(save_path, 'internal_pracid')):
    internal_df.write.parquet(os.path.join(save_path, 'internal_pracid'))

if not os.path.exists(os.path.join(save_path, 'external_pracid')):
    external_df.write.parquet(os.path.join(save_path, 'external_pracid'))



