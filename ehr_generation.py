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
from CPRD.functions import predictor_extractor
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
parser.add_argument("--disease", type=str, default='PD')
parser.add_argument("--cohort_dir", type=str, default='PD_data')
parser.add_argument("--seed", type=int, default=2024)

args = parser.parse_args()

cohort_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir)
save_path = os.path.join(cohort_path, 'EHR')
create_folder(save_path)

# read data
diagnoses_path = ''
medication_path = ''
procedure_path = ''
measure_path = ''
diagnoses = read_parquet(spark.sqlContext, diagnoses_path)
diagnoses = diagnoses.select(['patid', 'eventdate', 'ICD']).withColumnRenamed('ICD', 'code').dropna()
medications = read_parquet(spark.sqlContext, medication_path)
medications = medications.select(['patid', 'eventdate', 'code']).withColumnRenamed('bnfvtmcode', 'code').dropna()
procedure = read_parquet(spark.sqlContext, procedure_path)
procedure = procedure.select(['patid', 'eventdate', 'OPCS']).withColumnRenamed('OPCS', 'code').withColumn('code', F.concat(F.lit('PROC_'), F.col('code'))).dropna()
vals = read_parquet(spark.sqlContext, measure_path)
vals = vals[['patid','eventdate','snomed_mapped']].withColumnRenamed('snomed_mapped','code')
data = medications.union(procedure).union(diagnoses).union(vals).dropna()

cohort = read_parquet(spark.sqlContext, os.path.join(cohort_path, 'Cohort_CPRD')).withColumnRenamed('eventdate', f'{args.disease}_date').withColumn('label', F.lit(0))
cohort = cohort.select('patid', 'pracid', 'dob', f'{args.disease}_date', 'enddate', 'label')
lost_to_followup_records = cohort.select('patid', 'enddate').withColumn('code', F.lit('lost')).withColumnRenamed('enddate', 'eventdate')
data = data.unionByName(lost_to_followup_records)

behrt_formater = predictor_extractor.BEHRTextraction()
# col_entry means  when to stop getting records
pre_disease = behrt_formater.format_behrt(data, cohort, col_entry=f'{args.disease}_date', col_yob='dob', age_col_name='age', col_code='code', unique_in_months=6).dropna().drop('label').cache()
pre_disease_cohort = cohort.join(pre_disease, 'patid', 'inner').dropna()
pre_disease_cohort.write.parquet(os.path.join(save_path, f'ehr_b4_{args.disease}'))

# cut to internal and external
internal_pracid = read_parquet(spark.sqlContext, os.path.join(cohort_path, 'internal_pracid'))
external_pracid = read_parquet(spark.sqlContext, os.path.join(cohort_path, 'external_pracid'))
pre_disease_cohort = read_parquet(spark.sqlContext, os.path.join(save_path, f'ehr_b4_{args.disease}'))

from pyspark.sql.functions import monotonically_increasing_id

# Add row number to pre_disease_cohort
pre_disease_cohort = pre_disease_cohort.withColumn("row_num", monotonically_increasing_id())

# Joining operations with order by patid
internal_pre_disease_cohort = pre_disease_cohort.join(internal_pracid, 'pracid', 'inner').orderBy("patid")
external_pre_disease_cohort = pre_disease_cohort.join(external_pracid, 'pracid', 'inner').orderBy("patid")


if not os.path.exists(os.path.join(save_path, f'ehr_b4_{args.disease}_internal')):
    internal_pre_disease_cohort.write.parquet(os.path.join(save_path, f'ehr_b4_{args.disease}_internal'))
    external_pre_disease_cohort.write.parquet(os.path.join(save_path, f'ehr_b4_{args.disease}_external'))


