from Customize import JAVA_PATH, CPRD_CODE_PATH, COHORT_SAVE_PATH,MODEL_SAVE_PATH, get_symptom_medcode, get_icdcode

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
from pyspark.sql import Window

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
parser.add_argument("--experient_dir", type=str, default='AD')
parser.add_argument("--model_name", type=str, default='cl_maskage_b32')
parser.add_argument("--stage", type=str, default='before')
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--follow_up_year", type=int, default=5)
parser.add_argument("--k", type=int, default=5)

args = parser.parse_args()
stage = args.stage

cohort_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir)
ehr_path = os.path.join(cohort_path, 'EHR')
experient_dir = os.path.join(MODEL_SAVE_PATH, args.experient_dir)
model_save_dir = os.path.join(experient_dir, args.model_name+'_'+stage)
results_dir = os.path.join(model_save_dir, 'results')
results_save_dir = os.path.join(results_dir, f'{args.disease}-{args.k}')


demographics = read_parquet(spark.sqlContext, os.path.join(cohort_path, 'Cohort_CPRD')).withColumnRenamed('eventdate', f'{args.disease}_date')
demographics.show(5)


# endup follow
cohort = demographics.withColumn('endfollowupdate', F.col(f'{args.disease}_date') + F.expr(f'INTERVAL {args.follow_up_year} YEARS'))

# mortality
cohort_no_event = cohort.filter(F.col('dod').isNull()).withColumn('event', F.lit(0)).withColumn('time', F.least(F.col('enddate'), F.col('endfollowupdate')))
cohort_with_event = cohort.filter(F.col('dod').isNotNull()).filter(F.col('dod') > F.col(f'{args.disease}_date')).cache()
cohort_with_event_a = cohort_with_event.filter(F.col('dod') > F.col('endfollowupdate')).withColumn('event', F.lit(0)).withColumn('time', F.col('endfollowupdate'))
cohort_with_event_b = cohort_with_event.filter((F.col('dod') <= F.col('endfollowupdate')) & (F.col('dod') > F.col(f'{args.disease}_date'))).withColumn('event', F.lit(1)).withColumn('time', F.col('dod'))
cohort = cohort_no_event.union(cohort_with_event_a).union(cohort_with_event_b)
time2eventdiff = F.unix_timestamp('time', "yyyy-MM-dd") - F.unix_timestamp(f'{args.disease}_date', "yyyy-MM-dd")
cohort = cohort.withColumn('time', time2eventdiff).withColumn('time', (F.col('time') / 3600 / 24).cast('integer'))
# hospitalization
hes_path = ''
hes_raw_path = ''

# update based on zhengxian'code, adding condition about HES duration
hes = Diagnosis(read_txt(spark.sc, spark.sqlContext, hes_path)).cvt_admidate2date()
hes_duration = Diagnosis(read_txt(spark.sc, spark.sqlContext, hes_raw_path)).cvt_admidate2date()
hes = hes.join(hes_duration.select("patid", "spno", "duration"), on=["patid", "spno"], how="inner").filter(F.col("duration") >= 1)
hes = hes.withColumn("ICD", F.regexp_replace(F.col("ICD"), "\\.", ""))
disease_codes = get_icdcode(args.disease)
hes = hes.filter(F.col("ICD").isin(disease_codes))

hes_filtered = hes.join(cohort, 'patid', 'left').filter((F.col('admidate') <= F.col('endfollowupdate')) & (F.col('admidate') > F.col(f'{args.disease}_date'))).cache() # Find the earliest hospitalization date per patient
first_hospitalization = hes_filtered.groupBy('patid').agg(F.min('admidate').alias('admidate'))
cohort = cohort.join(first_hospitalization, 'patid', 'left')
# filtering
cohort_no_event = cohort.filter(F.col('admidate').isNull()).withColumn('event_hos', F.lit(0)).withColumn('time_hos', F.least(F.col('dod'), F.col('endfollowupdate')))
cohort_with_event = cohort.filter(F.col('admidate').isNotNull()).withColumn('event_hos', F.lit(1)).withColumn('time_hos', F.col('admidate'))
cohort = cohort_no_event.union(cohort_with_event)
time2eventdiff = F.unix_timestamp('time_hos', "yyyy-MM-dd") - F.unix_timestamp(f'{args.disease}_date', "yyyy-MM-dd")
cohort = cohort.withColumn('time_hos', time2eventdiff).withColumn('time_hos', (F.col('time_hos') / 3600 / 24).cast('integer'))


# write to file
cohort.toPandas().to_parquet(os.path.join(results_save_dir, 'demo_5y_mor_hos_sur_'+stage+'.parquet'))
demographics = read_parquet(spark.sqlContext, os.path.join(results_save_dir, 'demo_5y_mor_hos_sur_'+stage+'.parquet'))

import matplotlib.pyplot as plt
def plot_mor_hos_sur(result_df, results_df_name='internal'):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test


    kmf = KaplanMeierFitter()
    labels = sorted(result_df['label'].unique())
    fs_axis_labels = 20  # Font size for axis labels
    fs_legend = 18  # Font size for legend
    fs_ticks = 20  # Font size for ticks
    line_width = 2  # Line width for Kaplan-Meier plots
    fs_title = 20

    # Create a figure and axes for two subplots side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,8), dpi=150)

    for label in labels:
        cluster_data = result_df[result_df['label'] == label]
        kmf.fit(cluster_data['time'], event_observed=cluster_data['event'])
        ax[0].plot(kmf.survival_function_.index, 1 - kmf.survival_function_['KM_estimate'], label=f"Cluster {label}", linewidth=line_width)

    ax[0].set_xlabel('Time (days)', fontsize=fs_axis_labels)
    ax[0].set_ylabel('All-cause mortality', fontsize=fs_axis_labels)
    ax[0].tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax[0].set_title('Mortality', fontsize=fs_title)
    ax[0].grid(False)  # Grid removed

    # Time until first hospitalization plot
    for label in labels:
        cluster_data = result_df[result_df['label'] == label]
        kmf.fit(cluster_data['time_hos'], event_observed=cluster_data['event_hos'])
        ax[1].plot(kmf.survival_function_.index, 1 - kmf.survival_function_['KM_estimate'], label=f"Cluster {label}", linewidth=line_width)

    ax[1].set_xlabel('Time (days)', fontsize=fs_axis_labels)
    ax[1].set_ylabel('Hospitalisation', fontsize=fs_axis_labels)
    ax[1].tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax[1].set_title('Hospitalisation', fontsize=fs_title)
    ax[1].grid(False)  # Grid removed

    # Place the legend inside the right plot (hospitalization), removing the border
    handles, labels = ax[1].get_legend_handles_labels()  # Make sure handles and labels are collected from the correct subplot
    # ax[2].legend(handles, labels, fontsize=fs_legend, frameon=False) # loc='upper right'
    ax[1].legend(handles, labels, fontsize=fs_legend, frameon=False) # loc='upper right'
    plt.subplots_adjust(right=0.85)  # Adjust subplot parameters to fit the legend inside
    plt.savefig(os.path.join(results_save_dir, 'plot_mor_hos_sur_'+results_df_name+'.pdf'))
    plt.show()

def pie_chart(result_df, results_df_name='internal'):
    label_counts = result_df['label'].value_counts()
    # label_counts.columns = ['label', 'count']
    # Define colors from plt.cm.tab10
    colors = plt.cm.tab10.colors
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(result_df['label'].unique()))}
    # Now we plot the pie chart with the color mapping
    plt.figure(figsize=(6, 6), dpi=100)
    patches, texts, autotexts = plt.pie(
        label_counts.values,
        colors=[color_map[label] for label in label_counts.index ],
        autopct='%1.1f%%',
        startangle=140)
    # Adjust label positions to be further from the center and modify autotexts to include label, percentage, count.
    # Also make it bold, and move Cluster 2 a bit further away.
    for i, auto in enumerate(autotexts):
        auto.set_text(f'Cluster {label_counts.index[i]} \n {auto.get_text()}')
        auto.set_fontsize(15)
        auto.set_color('white')
        # auto.set_weight('bold')
        # Adjusting position to make labels further from center
        pos = auto.get_position()
        distance_factor = 1.45 if label_counts.index[i] == 1 else 1.1  # Move Cluster 2 further out
        adjusted_pos = (pos[0] * distance_factor, pos[1] * distance_factor)
        auto.set_position(adjusted_pos)
    # Ensure pie is drawn as a circle
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(results_save_dir, 'plot_pie_'+results_df_name+'.pdf'))
    # Show the pie chart
    plt.show()


cohort = pd.read_parquet(os.path.join(results_save_dir, 'demo_5y_mor_hos_sur_'+str(stage)+'_others.parquet'))
if stage =='before':
    internal_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_b4_{args.disease}_internal_with_label.parquet'))
    external_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_b4_{args.disease}_external_with_label.parquet'))
elif stage == 'after':
    internal_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_post_{args.disease}_internal_with_label.parquet'))
    external_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_post_{args.disease}_external_with_label.parquet'))
else:
    internal_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_all_{args.disease}_internal_with_label.parquet'))
    external_ehr = pd.read_parquet(os.path.join(results_save_dir, f'ehr_all_{args.disease}_external_with_label.parquet'))

# load data
external_eval_df = external_ehr.copy().drop(columns=['age',])
internal_eval_df = internal_ehr.copy().drop(columns=['age',])
non_overlap_cols = [col for col in cohort.columns if col not in external_eval_df.columns and col != 'patid']
external_result_df = external_eval_df.merge(cohort[['patid'] + non_overlap_cols], on='patid', how='inner')
internal_result_df = internal_eval_df.merge(cohort[['patid'] + non_overlap_cols], on='patid', how='inner')
mortality_rates = internal_result_df.groupby('label')['event'].mean().sort_values(ascending=True)  # Note the 'ascending=False' for descending order
reindex_mapping = {old_label: idx + 1 for idx, (old_label, _) in enumerate(mortality_rates.items())}
# calculate age
def calculate_age(result_df):
    result_df[f'{args.disease}_date'] = pd.to_datetime(result_df[f'{args.disease}_date'])
    result_df['dob'] = pd.to_datetime(result_df['dob'])
    result_df['age'] = (result_df[f'{args.disease}_date'] - result_df['dob'])/np.timedelta64(1, 'Y')
    result_df['label'] = result_df['label'].map(reindex_mapping) # reordering
    #print("original mortality rates", mortality_rates)
    return result_df

external_result_df = calculate_age(external_result_df)
internal_result_df = calculate_age(internal_result_df)
plot_mor_hos_sur(external_result_df, 'external')
plot_mor_hos_sur(internal_result_df, 'internal')

pie_chart(external_result_df, 'external')
pie_chart(internal_result_df, 'internal')

from IPython.display import HTML

# transfer result_df for visualization
def safe_int_convert(x):
    try:
        return int(x)
    except ValueError:
        return np.nan
    
def create_and_concat_dummies(df, col_name, prefix=None):
    dummies = pd.get_dummies(df[col_name], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    return df

# Function to calculate mean and standard deviation
def calculate_mean(series):
    return f"{series.mean():.2f} ({series.std():.2f}) Miss({series.isna().sum() / len(series) * 100 :.1f}%)"

def calculate_percentage_and_count(series):
    count_ones = (series == 1).sum()
    total_count = len(series)
    percentage = (count_ones / total_count) * 100
    return f"{count_ones} ({percentage:.1f}%) Miss({series.isna().sum() / len(series):.1f}%)"

def first(series):
    return series.iloc[0]
    # return series[0]

def df_format(result_df):
    smoke_categories =  {1: 'Smoker', 2: 'Ex-Smoker', 3: 'Non-Smoker'}
    gender_categories = {1: 'Male', 2: 'Female'}
    # bmi_categories = {1: 'Quintile 1', 2: 'Quintile 2', 3: 'Quintile 3', 4: 'Quintile 4', 5: 'Quintile 5'}
    region_categories = {1: 'North East',2: 'North West',3: 'Yorkshire And The Humber',4: 'East Midlands',5: 'West Midlands',
        6: 'East of England',7: 'South West', 8: 'South Central',9: 'London',10: 'South East Coast',11: 'Northern Ireland',12: 'Scotland',13: 'Wales'}

    result_df['smoke'] = result_df['smoke'].map(smoke_categories)
    result_df['gender'] = result_df['gender'].apply(safe_int_convert).astype('Int64').map(gender_categories)
    result_df['bmi_cat'] = pd.cut(result_df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['BMI underweight', 'BMI healthy', 'BMI overweight', 'BMI obesity',])
    result_df['region'] = result_df['region'].apply(safe_int_convert).astype('Int64').map(region_categories)
    result_df = create_and_concat_dummies(result_df, 'smoke',)
    result_df = create_and_concat_dummies(result_df, 'gender', )
    result_df = create_and_concat_dummies(result_df, 'bmi_cat', )
    result_df = create_and_concat_dummies(result_df, 'region', 'region')
    result_df = create_and_concat_dummies(result_df, 'imd2015_5', 'imd')
    result_df = create_and_concat_dummies(result_df, 'gen_ethnicity', 'ethnicity')

    copy_df = result_df.copy()
    result_df = copy_df.copy()
    print(result_df.columns)
    result_df = copy_df.copy()

    continuous_vars = ['age', 'sbp',  'bmi'] # 'systolic',  'bmi'
    categorical_vars = ['Male', 'Female', 
                        # 'Fall', 'Language', 'Dementia', 'AD_Dementia', 'Depression', 'Anxiety', 'Memory', 'Constipation', 'FOG', 'Sleep',
                        'BMI underweight', 'BMI healthy', 'BMI overweight', 'BMI obesity', 'Smoker', 'Ex-Smoker', 'Non-Smoker','ethnicity_White',
                        'imd_1', 'imd_2', 'imd_3', 'imd_4','imd_5']                    
    result_df = result_df[categorical_vars + continuous_vars + ['label'] ]
    cluster_sizes = result_df['label'].value_counts()
    result_df['Cluster'] = result_df['label']
    result_df['N'] = result_df['label'].map(cluster_sizes).apply(lambda x: f"{x:,}")

    aggregation_dict = {}
    # Add mean and std calculation for continuous variables
    for col in continuous_vars:
        aggregation_dict[col] = calculate_mean
    for col in categorical_vars:
        aggregation_dict[col] = calculate_percentage_and_count
    aggregation_dict['N'] = first
    # Group by 'label' and apply the aggregation
    grouped_df = result_df.groupby('Cluster').agg(aggregation_dict)
    formatted_df = grouped_df.T
    return formatted_df, copy_df

# # result_df
formatted_interal_df, copy_internal = df_format(internal_result_df)
formatted_external_df, copy_external  = df_format(external_result_df)
formatted_interal_df.to_csv(os.path.join(results_save_dir, 'formatted_df_internal.csv'))
formatted_external_df.to_csv(os.path.join(results_save_dir, 'formatted_df_external.csv'))




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import icd10

def process_codes(code_list):
    processed_codes = set()
    for code in code_list:
        # if code.startswith("VAL_"):
        #     continue
        if len(code) == 4:
            processed_code = code[:3]
        elif code.startswith('PROC_'):
            processed_code = code[:8]
        elif code == 'SEP':
            continue
        elif code.startswith('bnf_'):
            processed_code = code[:8]
        elif code.startswith('vtm_'):
            processed_code = code
        else:
            processed_code = code
        processed_codes.add(processed_code)
    return list(processed_codes)


def map_to_target(code):
    if pd.notna(code):
        # Custom mappings for special cases
        special_cases = {
            'Other chronic obstructive pulmonary disease': 'COPD',
        }
        # print(code)
        # Check if the code is in the special cases
        if code in special_cases:
            return special_cases[code]
        # Check if the code is in the label_to_target_map
        elif code in label_to_target_map:
            return label_to_target_map[code]
        # If not in the map, use icd10 to find the description
        else:
            try:
                return icd10.find(code).description
            except:
                return code  # Return the code itself if not found in icd10
    else:
        return code


def encod_df(copy_internal, df_label='internal'):
    if not os.path.exists( os.path.join(results_save_dir, 'one_hot_encoded_'+stage+'_EHR_'+df_label+'.pkl') ):
        result_df = copy_internal.copy()
        # Combine both sections of code processing
        result_df['processed_code'] = result_df['code'].apply(process_codes)
        # Flatten the list of processed codes and get unique codes
        all_codes = set(code for sublist in result_df['processed_code'] for code in sublist)
        # One-hot encode the codes efficiently
        one_hot_encoded = pd.DataFrame({code: result_df['processed_code'].apply(lambda x: code in x).astype(int) for code in all_codes})
        # write to file
        result_df.to_pickle(os.path.join(results_save_dir, 'result_df_'+str(stage)+'_EHR'+'.pkl'))
        one_hot_encoded.to_pickle(os.path.join(results_save_dir, 'one_hot_encoded_'+str(stage)+'_EHR_'+df_label+'.pkl'))

    else:
        result_df = pd.read_pickle(os.path.join(results_save_dir, 'result_df_'+str(stage)+'_EHR'+'.pkl'))
        one_hot_encoded = pd.read_pickle(os.path.join(results_save_dir, 'one_hot_encoded_' +str(stage)+'_EHR_'+df_label+'.pkl'))
    # Calculate the range of proportions for each code
    proportion_ranges = one_hot_encoded.groupby(result_df['label']).mean().apply(lambda x: x.max() - x.min(), axis=0)
    return proportion_ranges, one_hot_encoded, result_df

code_map = pd.read_csv('/home/wfan/Desktop/CPRD_Cut22/CBEHRT/general_model_newCutCPRD/Data/allcode_description.csv')
code_map['source'] = code_map['source'].str.replace(r'^DIA_|^MED_', '', regex=True)
code_map['source'] = code_map['source'].apply(lambda x: x[:8] if x.startswith('bnf_') else x)
# Your code for calculating proportion_ranges and significant_codes here...
label_to_target_map = dict(zip(code_map['source'], code_map['target']))



def plot_heat_map(result_df, proportion_ranges,one_hot_encoded, plot_code='icd', variation_threshold=0.15, df_label ='internal'):
    #variation_threshold = 0.15 # Adjust as needed
    significant_codes = proportion_ranges[proportion_ranges > variation_threshold].index.tolist()
    # if 'E14' in significant_codes:  
    #     significant_codes.remove('E14')
    significant_codes = pd.Series(significant_codes)
    if plot_code == 'icd':
        significant_codes = significant_codes[significant_codes.apply(len)==3]
    elif plot_code == 'proc':
        significant_codes = significant_codes[significant_codes.apply(lambda x: x.startswith('PROC_'))]
    elif plot_code == 'med':
        significant_codes = significant_codes[significant_codes.apply(lambda x: x.startswith('bnf_') or x.startswith('vtm_'))]
    elif plot_code == 'val':
        significant_codes = significant_codes[significant_codes.apply(lambda x: x.startswith('VAL_'))]
    elif plot_code == 'all':
        significant_codes = significant_codes
    else:
        assert plot_code in ['icd','proc','med','val','all']

    significant_codes = pd.Index(sorted(significant_codes))
    co_matrix = one_hot_encoded[significant_codes].groupby(result_df['label']).mean().T
    co_matrix.index = co_matrix.index.map(map_to_target)
    co_matrix.index = co_matrix.index.to_series().replace({'Other chronic obstructive pulmonary disease': 'COPD'})
    co_matrix = co_matrix.rename_axis(index='target')
    title_fontsize = 20
    label_fontsize = 25
    tick_fontsize = 15
    annotation_fontsize = 15
    # Plot the heatmap with larger font sizes
    plt.figure(figsize=(45, len(significant_codes) / 2), dpi=100)
    sns.heatmap(co_matrix, cmap='Blues', annot=True, fmt=".2f", annot_kws={'size': annotation_fontsize})
    plt.xlabel('Cluster', fontsize=label_fontsize)
    # plt.ylabel('Significant Codes', fontsize=label_fontsize)
    y_tick_labels = [label.title() if not label.isupper() else label for label in co_matrix.index]
    y_tick_labels = [label[:50] for label in y_tick_labels]
    plt.yticks(ticks=np.arange(len(co_matrix.index)) + 0.5, labels=y_tick_labels, fontsize=tick_fontsize, rotation=0, ha='right')
    plt.xticks(ticks=np.arange(len(co_matrix.columns)) + 0.5, fontsize=tick_fontsize, rotation=0, ha='center')
    plt.savefig(os.path.join(results_save_dir,'heatmap', f'heatmap_{stage}_{plot_code}_thred_{variation_threshold}_{df_label}.pdf'))
    plt.show()

internal_proportion_ranges, one_hot_encoded_internal, result_df_internal = encod_df(copy_internal, 'internal')
plot_heat_map(result_df_internal, internal_proportion_ranges, one_hot_encoded_internal, 'all', 0.15)
plot_heat_map(result_df_internal, internal_proportion_ranges, one_hot_encoded_internal, 'icd', 0.15)
plot_heat_map(result_df_internal, internal_proportion_ranges, one_hot_encoded_internal, 'med', 0.15)
plot_heat_map(result_df_internal, internal_proportion_ranges, one_hot_encoded_internal, 'val', 0.15)
plot_heat_map(result_df_internal, internal_proportion_ranges, one_hot_encoded_internal, 'proc', 0.15)

external_proportion_ranges, one_hot_encoded_external, result_df_external = encod_df(copy_external, 'external')
plot_heat_map(result_df_external, external_proportion_ranges, one_hot_encoded_external, 'all', 0.15, 'external')
plot_heat_map(result_df_external, external_proportion_ranges, one_hot_encoded_external, 'icd', 0.15, 'external')
plot_heat_map(result_df_external, external_proportion_ranges, one_hot_encoded_external, 'med', 0.15, 'external')
plot_heat_map(result_df_external, external_proportion_ranges, one_hot_encoded_external, 'val', 0.15, 'external')
plot_heat_map(result_df_external, external_proportion_ranges, one_hot_encoded_external, 'proc', 0.15, 'external')
