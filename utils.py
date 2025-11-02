import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime, date

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

import shap
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def calculate_age(dob, admit_time):
  """
  Calculates and returns age
  
  Input: Date of Birth and Admission Time
  Output: Age in years
  """
  
  age = datetime.strptime(admit_time, '%Y-%m-%d %H:%M:%S').date() - datetime.strptime(dob, '%Y-%m-%d %H:%M:%S').date()
  return age.days // 365


def time_difference(a, b):
  """
  Calculates the difference between two timestamps in hours
  
  Input: Timestamps
  Output: Time difference in hours
  """
  
  diff = datetime.strptime(a, '%Y-%m-%d %H:%M:%S').date() - datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
  return diff.total_seconds() / 3600


def encode_race(df):
  """
  Encode race from df to numerical representation

  Input: Pandas DataFrame (This dataframe is the output from fetch_demographic(), it is necessary to call it before calling this function)
  Output: Pandas DataFrame with encoded race
  """
  
  df['WHITE'] = df['race'].apply(lambda x: 1 if x == 'WHITE' else 0)
  df['HISPANIC'] = df['race'].apply(lambda x: 1 if 'hispanic' in x.lower() else 0)
  df['BLACK/AFRICAN AMERICAN'] = df['race'].apply(lambda x: 1 if 'black' in x.lower() else 0)
  df['ASIAN'] = df['race'].apply(lambda x: 1 if 'asian' in x.lower() else 0)
  df['OTHER'] = df['race'].apply(lambda x: 1 if x in ['UNKNOWN/NOT SPECIFIED', 'OTHER', 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE', 'UNABLE TO OBTAIN'] else 0)
  return df


def aki_stage(creatinine):
  """
  Takes in creatinine value and returns the creatinine class

  Input: Creatinine value
  Output: Creatinine class
  """

  if creatinine <= 1.2:
      return "normal"
  elif creatinine <= 2.0:
      return "mild"
  else:
      return "severe"

def encode_class(value):

  """
  Takes in creatinine class and encodes it to a number. Before calling this function it is absolutely necessary to call aki_stage() function on creatinine values.

  Input: Creatinine class
  Output: Creatinine class encoded
  """
    
  if value == 'mild':
      return 1
  elif value == 'normal':
      return 0
  else:
      return 2

def evaluate_model(model, X_test, y_test):

  """
    Evaluates a trained classification model on test data and prints key metrics.

    Computes and prints:
    - Accuracy
    - Macro and micro AUROC (One-vs-Rest)
    - Full classification report (precision, recall, F1-score per class)

    Input:
        model: Trained classification model.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): True labels for the test set.

    Output:
        None
    """


  y_pred_proba = model.predict_proba(X_test)

  lb = LabelBinarizer()
  y_test_binarized = lb.fit_transform(y_test)
    
  print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 4))

  macro_roc_auc_ovr = roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr", average="macro")
  print(f"Macro AUROC (OvR): {macro_roc_auc_ovr:.4f}")
    
  micro_roc_auc_ovr = roc_auc_score(y_test_binarized, y_pred_proba, multi_class="ovr", average="micro")
  print(f"Micro AUROC (OvR): {micro_roc_auc_ovr:.4f}")
    
  print("=================================================")
  print("Classification Report:")
  print(classification_report(y_test, model.predict(X_test), digits=4))

def shap_results(model, X_test, all_items, X_train=None):
    
    """
    Compute and visualize SHAP feature importances for a tree-based model by aggregating over instances and over classes. 

    Input:
        model: Trained tree-based model (e.g., XGBoost, LightGBM, or sklearn tree ensemble).
        X_test (pd.DataFrame): Test dataset features.
        all_items (dict): Dictionary mapping feature (labs + vitals) indices or codes to descriptive names.
        X_train (pd.DataFrame, optional): Training dataset features (used by SHAP if needed). Default is None.

    Output:
        None
    Prints:
        - Top 5 features with their corresponding descriptions.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  
    
    values = np.stack(shap_values, axis=0)
    values = values.mean(axis=0)
    shap_importance = np.abs(values).mean(axis=0)

    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": shap_importance
    }).sort_values(by="importance", ascending=False)

    # all_items = {**lab_mapper, **chart_mapper}
    renamed = dict()

    for i in range(5):
        v = feature_importance.iloc[i]['feature']
        k = v
        try:
            split = feature_importance.iloc[i]['feature'].split('_')
            v = f"{all_items[int(split[0])]}_{'_'.join(split[1:])}"
        except: 
            pass
        renamed[k] = v


    print(f"Top Features: ")
    print(pd.DataFrame({'Features': list(renamed.keys()),
                       'Description': list(renamed.values())}))

    shap.summary_plot(shap_values, X_test, show=False, plot_type='bar')
    plt.show()


def load_data(file_path):
  '''
  Fetch the file present at the path

  Input: File path
  Output: Pandas DataFrame preset at that path
  '''
  
  data = pd.read_csv(file_path)
  return data

def fetch_demographic(patients, admissions, icustays):
  """
    Fetch demographic data for each patient.

    Loops over `subject_id` values to build a demographic profile for each patient.  
    Only includes patients with age < 100.

    Input:
        patients (pd.DataFrame): Patient-level data (e.g., age, gender, DOB).
        admissions (pd.DataFrame): Hospital admissions data.
        icustays (pd.DataFrame): ICU stay data.

    Output:
        pd.DataFrame: Demographic data for patients with age < 100.
  """
    
  hadms, stay_ids, sex, age = [], [], [], [] 
  pa_ids = patients['subject_id'].tolist()
    
  demo_df = pd.DataFrame(columns=['subject_id', 'hadm_id', 'race', 'gender', 'age',
                                   'admission_type'])
    
  for i in pa_ids:
        
      curr_patient = patients[patients['subject_id'] == i]
      curr_admission = admissions[admissions['subject_id'] == i]
      curr_icustay = icustays[icustays['subject_id'] == i]

      curr_info = pd.DataFrame({'subject_id': [i] * len(curr_admission),
                                'hadm_id': curr_admission['hadm_id'].tolist(),
                                  'race': [admissions[admissions['subject_id'] == i]['ethnicity'].tolist()[0]] * len(curr_admission),
                                'age': [calculate_age(curr_patient['dob'].tolist()[0], min(curr_admission['admittime'].tolist()))] * len(curr_admission),
                                'gender': [curr_patient['gender'].tolist()[0]] * len(curr_admission),
                                'admittime': curr_admission['admittime'].tolist(),
                                'dischtime': curr_admission['dischtime'].tolist(),
                                'admission_type': curr_admission['admission_type'].tolist()})
        
      demo_df = pd.concat([demo_df, curr_info], axis=0)

  return demo_df[demo_df['age'] < 100]


def fetch_labs(d_labs, labevents):

  """
    Fetch lab data relevant to creatinine.

    Extracts lab measurements from `labevents` that are necessary
    for or may affect creatinine. Also builds a mapping
    between lab item IDs and their descriptive names.

    Input:
        labevents (pd.DataFrame): Lab events DataFrame (raw lab results).
        d_labitems (pd.DataFrame): Lab items dictionary DataFrame containing
            metadata and descriptions of lab tests.

    Output:
        tuple:
            pd.DataFrame: Lab data specific to creatinine.
            dict: Mapping of lab item IDs to their descriptive names.
    """
  
  # Labs that creatinine requires
  target_labs = [
    "creatinine",
    "urea nitrogen", "bun",
    "sodium", "potassium", "chloride",
    "bicarbonate", "anion gap",
    "calcium", "magnesium", "phosphate",
    "glucose", "albumin", "lactate",
    "hemoglobin", "hematocrit",
    "white blood cell", "wbc",
    "platelet", "bilirubin", 'ptt', 'rbc', 'red blood cell', 'prothrombin', 'partial thromboplastin', 'inr']
    
  need_labs = []

  filtered_dlabs = d_labs[d_labs['fluid'].isin(['BLOOD', 'Blood'])]
  for i in target_labs:
       for j in filtered_dlabs['label'].unique():
          if i in j.lower():
              need_labs.append(j)
    
    
  lab_items = d_labs[d_labs['label'].isin(need_labs)]['itemid'].tolist()
  lab_mapper = {i: d_labs[d_labs['itemid'] == i]['label'].tolist()[0] for i in lab_items}
    
  return labevents[labevents['itemid'].isin(lab_items)], lab_mapper


def fetch_chartevents(d_items, chartevents):

  """
    Fetch vital data relevant to creatinine.

    Extracts vital sign measurements from `chartevents` that are necessary
    for or may affect creatinine. Also builds a mapping
    between item IDs and their descriptive names.

    Input:
        chartevents (pd.DataFrame): Chart events DataFrame (raw vital data).
        d_items (pd.DataFrame): Items dictionary DataFrame containing metadata
            and descriptions of charted variables.

    Output:
        tuple:
            pd.DataFrame: Vital data specific to creatinine.
            dict: Mapping of vital item IDs to their descriptive names.
    """
    
  vitals = ["heart rate", "respiratory rate", "temperature", "spo2", "blood pressure"]
    
  vital_categories = [
    'Routine Vital Signs',
    'Hemodynamics',
    'Respiratory',
    'Cardiovascular',
    'General',
    'Pulmonary']

  d_items_vitals = d_items[d_items['category'].isin(vital_categories)]
    
  need_vitals = ['SpO2']
  org_ditems = [str(i) for i in d_items_vitals['label']]

  for i in vitals:
      for j in org_ditems:
          if i.lower() in j.lower():
              need_vitals.append(j)
    
  chart_items = d_items[d_items['label'].isin(need_vitals)]['itemid'].tolist()
  chart_mapper = {i: d_items[d_items['itemid'] == i]['label'].tolist()[0] for i in chart_items}
  return chartevents[chartevents['itemid'].isin(chart_items)], chart_mapper


def preprocess_events(creat_labevents, creat_chartevents):

  """
    Preprocess creatinine lab and chart (vital) events for better processing.

    This includes handling NaN values, processing value units, and removing
    unnecessary columns.

    Input:
        creatinine labevents (pd.DataFrame): Creatinine lab events DataFrame.
            Output of `fetch_labdata()`.
        creatinine chartevents (pd.DataFrame): Chart (vital) events DataFrame.
            Output of `fetch_chartevents()`.

    Output:
        tuple:
            pd.DataFrame: Preprocessed creatinine lab events.
            pd.DataFrame: Preprocessed creatinine chart (vital) events.
   """
    
  creat_chartevents.dropna(subset = 'valuenum', inplace = True)
  creat_labevents.dropna(subset = 'valuenum', inplace = True)
  creat_labevents.dropna(subset = 'valueuom', inplace = True)
    
  creat_chartevents.loc[creat_chartevents['valueuom'] == '?C', 'value'] = creat_chartevents.loc[creat_chartevents['valueuom'] == '?C', 'value'].apply(lambda x: x * 9/5 + 32)
  creat_chartevents.loc[creat_chartevents['valueuom'] == '?C', 'valueuom'] = '?F'
    # creat_chartevents.loc[creat_chartevents['vital_name'] == 'Temperature Celsius', 'vital_name'] = 'Temperature Fahrenheit'
  creat_chartevents.loc[creat_chartevents['valueuom'] == 'bpm', 'valueuom'] = 'BPM'
    
  return creat_labevents, creat_chartevents


def get_static_data(hadm_id, demo_df):

  """
    Extracts static demographics for a given hospital stay.

    Given a hospital admission ID (`hadm_id`), returns a single-row DataFrame
    containing the static features for that admission.

    Args:
        hadm_id (int): Hospital admission ID to extract data for.
        demo_df (pandas.DataFrame): Demographic/admission DataFrame containing
            features for all admissions.

    Returns:
        pandas.DataFrame: Single-row DataFrame with static features for the specified admission.
  """
  
  curr_demo = demo_df[demo_df['hadm_id'] == hadm_id].drop(['subject_id', 'hadm_id', 'gender', 'admittime', 'dischtime','race'], axis=1)
  return pd.DataFrame(curr_demo.iloc[0]).T

def get_nowcastinglabs(creat_labevents, demo_df, lab_mapper):

  """
    Construct structured lab event features for nowcasting task.

    For each hospital admission (`hadm_id`), extracts creatinine-related lab
    measurements and computes features such as time since admit,
    time since last lab, last lab value, number of tests so far, and abnormal
    flags. The result is a dictionary mapping each admission ID to a
    DataFrame of per-item lab features.

    Input:
        creat_labevents (pd.DataFrame): Lab events DataFrame filtered for
            creatinine-related measurements. Output of `preprocess_events()`.
        demo_df (pd.DataFrame): Demographic DataFrame containing
            'hadm_id' and corresponding 'admittime'.
        lab_mapper (dict): Mapping of lab item IDs to their descriptive names

    Output:
        dict:
            Dictionary of keys hospital admission IDs (`hadm_id`) and
            values of pandas DataFrames containing explicit lab features
            for each item ID. Each DataFrame includes columns such
            as:
                - `{itemid}_time_since_admit`
                - `{itemid}_time_since_last_lab`
                - `{itemid}_last_lab_value`
                - `{itemid}_valuenum`
                - `{itemid}_num_tests_so_far`
                - `{itemid}_abnormal_flag`
    """
    
  hadm_ids = creat_labevents['hadm_id'].unique()
  items = list(lab_mapper.keys())
  admittimes = {i: demo_df[demo_df['hadm_id'] == i]['admittime'].tolist()[0] for i in demo_df['hadm_id'].tolist()}
    
  lab_dict = dict()
    
  for i in hadm_ids:
        
      curr_lab = creat_labevents[creat_labevents['hadm_id'] == i].sort_values(by='charttime')
      working_lab = curr_lab[['itemid', 'charttime', 'valuenum', 'flag']]
        
      working_lab["last_lab_value"] = working_lab.groupby(["itemid"])["valuenum"].shift(1)
        
      admittime = admittimes[i]
      working_lab["charttime"] = pd.to_datetime(working_lab["charttime"])
      working_lab["time_since_admit"] = abs((working_lab["charttime"] - pd.to_datetime(admittime)).dt.total_seconds() / 3600.0)
        
      working_lab["time_since_last_lab"] = working_lab.groupby(["itemid"])["charttime"].diff().dt.total_seconds() / 3600.0
        
      working_lab["time_since_last_lab"] = working_lab["time_since_last_lab"].fillna(0.0)
      working_lab["last_lab_value"] = working_lab["last_lab_value"].fillna(-1)
      working_lab["num_labs_so_far"] = working_lab.groupby(["itemid"]).cumcount() + 1
        
      working_lab['flag'] = working_lab['flag'].apply(lambda x: 1 if x == 'abnormal' else 0)
        
        
      # print("Collecting Items...")
        
      items_df = []
      for j in items:
            
          curr_item = working_lab[working_lab['itemid'] == j]
          items_df.append(pd.DataFrame({f'{j}_time_since_admit': curr_item['time_since_admit'].tolist(),
                                         f'{j}_time_since_last_lab': curr_item['time_since_last_lab'].tolist(),
                                         f'{j}_last_lab_value': curr_item['last_lab_value'].tolist(),
                                         f'{j}_valuenum': curr_item['valuenum'].tolist(),
                                          f'{j}_num_tests_so_far': curr_item['num_labs_so_far'].tolist(),
                                          f'{j}_abnormal_flag': curr_item['flag'].tolist()
                                         }))
        
        
      lab_dict[i] = pd.concat(items_df, axis=1)
        
  return lab_dict
    

def get_nowcastingvitals(creat_chartevents, demo_df, chart_mapper):

  """
    Construct structured vital event features for nowcasting task.

    For each hospital admission (`hadm_id`), extracts creatinine-related vital
    measurements and computes features such as time since admit,
    time since last vital, last vital value, number of tests so far, and abnormal
    flags. The result is a dictionary mapping each admission ID to a
    DataFrame of per-item vital features.

    Input:
        creat_chartevents (pd.DataFrame): Chart events DataFrame filtered for
            creatinine-related measurements. Output of `preprocess_events()`.
        demo_df (pd.DataFrame): Demographic DataFrame containing
            'hadm_id' and corresponding 'admittime'.
        chart_mapper (dict): Mapping of lab item IDs to their descriptive names

    Output:
        dict:
            Dictionary of keys hospital admission IDs (`hadm_id`) and
            values of pandas DataFrames containing explicit vital features
            for each item ID. Each DataFrame includes columns such
            as:
                - `{itemid}_time_since_admit`
                - `{itemid}_time_since_last_vital`
                - `{itemid}_last_vital_value`
                - `{itemid}_valuenum`
                - `{itemid}_num_tests_so_far`
                - `{itemid}_abnormal_flag`
    """
    
  hadm_ids = creat_chartevents['hadm_id'].unique()
  items = list(chart_mapper.keys())
  admittimes = {i: demo_df[demo_df['hadm_id'] == i]['admittime'].tolist()[0] for i in demo_df['hadm_id'].tolist()}
    
  vital_dict = dict()
    
  for i in hadm_ids:
        
      curr_vital = creat_chartevents[creat_chartevents['hadm_id'] == i].sort_values(by='charttime')
      working_vital = curr_vital[['itemid', 'charttime', 'valuenum']]
        
      working_vital["last_vital_value"] = working_vital.groupby(["itemid"])["valuenum"].shift(1)
        
      admittime = admittimes[i]
      working_vital["charttime"] = pd.to_datetime(working_vital["charttime"])
      working_vital["time_since_admit"] = abs((working_vital["charttime"] - pd.to_datetime(admittime)).dt.total_seconds() / 3600.0)
        
      working_vital["time_since_last_vital"] = working_vital.groupby(["itemid"])["charttime"].diff().dt.total_seconds() / 3600.0
        
      working_vital["time_since_last_vital"] = working_vital["time_since_last_vital"].fillna(0.0)
      working_vital["last_vital_value"] = working_vital["last_vital_value"].fillna(-1)
      working_vital["num_vitals_so_far"] = working_vital.groupby(["itemid"]).cumcount() + 1
        
        
        # print("Collecting Items...")
        
      items_df = []
      for j in items:
            
          curr_item = working_vital[working_vital['itemid'] == j].sort_values(by='charttime')
          items_df.append(pd.DataFrame({f'{j}_time_since_admit': curr_item['time_since_admit'].tolist(),
                                         f'{j}_time_since_last_vital': curr_item['time_since_last_vital'].tolist(),
                                         f'{j}_last_vital_value': curr_item['last_vital_value'].tolist(),
                                         f'{j}_valuenum': curr_item['valuenum'].tolist(),
                                          f'{j}_num_tests_so_far': curr_item['num_vitals_so_far'].tolist(),
                                         }))
        
      vital_dict[i] = pd.concat(items_df, axis=1)
        
  return vital_dict



def get_nowcastingdata(lab_dict, vital_dict, demo_df):

  """
    Combines lab features, vital features, and demographic/static data into a single (final) dataset.

    For each hospital admission (hadm_id), merges the corresponding lab and vital
    time series (from `lab_dict` and `vital_dict`) with encoded demographic and
    static admission data (from `demo_df`). Admissions with missing or empty
    lab/vital data are skipped.

    Input:
        lab_dict (dict): 
            Dictionary mapping hadm_id to a DataFrame of explicitly formed lab features (get_nowcastinglab()).
        vital_dict (dict): 
            Dictionary mapping hadm_id to a DataFrame of explicitly formed vital features (get_nowcastingvital()).
        demo_df (pandas.DataFrame): 
            Demographic data (includes gender, race, admission type, etc.).

    Output:
        pandas.DataFrame: 
            Combined dataset of lab, vital, and demographic features across all admissions.
  """
    
  demo_df['sex'] = demo_df['gender'].apply(lambda x: 1 if x == 'F' else 0)
  demo_df = encode_race(demo_df)
  encoded_demo = pd.get_dummies(demo_df, columns=['admission_type'], prefix='admit', dtype=int)
      
  full_data = []
      
  hadms = list(lab_dict.keys())
  for i in hadms:
      if (i not in list(vital_dict.keys())) or (0 in [len(lab_dict[i]), len(vital_dict[i])]):
          pass
      else:
          combined = pd.concat([lab_dict[i], vital_dict[i]], axis=1)
          static_df = get_static_data(i, encoded_demo)
            
          for i in static_df.columns:
                combined[i] = static_df[i].tolist() * len(combined)
          
          full_data.append(combined)
  return pd.concat(full_data, axis=0)


def process_data(full_data, lab_mapper):

  """
    Preprocesses and generates labels for creatinine prediction.

    This function:
    - Selects the creatinine lab column based on `lab_mapper`.
    - Drops rows with missing creatinine values. Imputation would induce wrong or irrelevent values. 
    (Example: Creatinine value stayed the same eventhough other lab/vital value changed)
    - Fills remaining missing values with -1.
    - Removes columns with only a single unique value.
    - Computes Acute Kidney Injury (AKI) stage labels and encodes them as integers.
    - Separates features (X) and labels (y) for modeling.

    Input:
        full_data (pandas.DataFrame): 
            Combined dataset containing lab, vital, and demographic features.
        lab_mapper (dict): 
            Dictionary mapping lab item IDs to their descriptive names.

    Output:
        tuple:
            X (pandas.DataFrame): Features for modeling (all columns except creatinine label and raw value column).
            y (pandas.Series): Encoded creatinine class labels.
    """
    
  label_col = [list(lab_mapper.keys())[list(lab_mapper.values()).index(i)] for i in list(lab_mapper.values()) if 'creatinine' in i.lower()]
  creat_df = full_data.dropna(subset=[f'{label_col[0]}_valuenum'], axis=0)
    
  creat_df.fillna(-1, inplace=True)
  cols = []
  for i in creat_df.columns:
      if len(creat_df[i].unique()) < 2:
          cols.append(i)
    
  creat_df.drop(cols, axis=1, inplace=True)
  creat_df["creat_class"] = creat_df[f'{label_col[0]}_valuenum'].apply(aki_stage)
  creat_df['creat_class'] = creat_df['creat_class'].apply(encode_class)
    
  X = creat_df.drop(['creat_class', '50912_valuenum'], axis=1)
  y = creat_df['creat_class']

  return X, y

def get_loaders(X_train, y_train, X_test, y_test):
    
    """
    Creates PyTorch DataLoaders for training, validation, and testing from pandas DataFrames/Series.

    The training set is split into training and validation subsets (80%-20% split). Features are converted to float32 and labels to long. Batch Size used is 64

    Input:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series or pd.DataFrame): Test labels.

    Output:
        tuple: (train_loader, val_loader, test_loader)
            - train_loader (DataLoader): DataLoader for the training subset, shuffled.
            - val_loader (DataLoader): DataLoader for the validation subset, not shuffled.
            - test_loader (DataLoader): DataLoader for the test set, not shuffled.
    """
    
    n = len(X_train)
    train_size = int(0.8 * n)
    val_size   = n - train_size 

    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    
    return train_loader, val_loader, test_loader


class NN(nn.Module):
  """
    A fully connected feedforward neural network with two hidden layers, batch normalization, dropout, and ReLU activations.

    Architecture:
        - Input layer of size `input_dim`
        - Hidden layer 1: Linear -> BatchNorm -> ReLU -> Dropout (0.3 probability)
        - Hidden layer 2: Linear -> BatchNorm -> ReLU -> Dropout (0.3 probability)
        - Output layer: Linear (size `output_dim`)

    Input:
        input_dim (int): Number of input features. Default is 321.
        hidden_dim (int): Number of neurons in the first hidden layer. Default is 256.
        hidden_dim2 (int): Number of neurons in the second hidden layer. Default is 128.
        output_dim (int): Number of output neurons (number of classes). Default is 3.

    Forward pass:
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # raw logits
    """

  def __init__(self, input_dim=321, hidden_dim=256, hidden_dim2=128, output_dim=3):
      super(NN, self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim)
      self.bn1 = nn.BatchNorm1d(hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
      self.bn2 = nn.BatchNorm1d(hidden_dim2)
      self.fc3 = nn.Linear(hidden_dim2, output_dim)
      self.dropout = nn.Dropout(0.3)
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.bn1(self.fc1(x)))
      x = self.dropout(x)
      x = self.relu(self.bn2(self.fc2(x)))
      x = self.dropout(x)
      x = self.fc3(x) 
      return x
  
def train_nn(model, train_loader, val_loader, epochs=10):

  """
    Train a PyTorch neural network on the given training and validation data loaders.

    Input:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train. Default is 10.

    Output:
        tuple: (train_acc_l, val_acc_l, train_loss_l, val_loss_l)
            - train_acc_l: List of training accuracies per epoch.
            - val_acc_l: List of validation accuracies per epoch.
            - train_loss_l: List of training losses per epoch.
            - val_loss_l: List of validation losses per epoch.
    """

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  train_acc_l, train_loss_l, val_acc_l, val_loss_l = [], [], [], []
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(epochs):
      model.train()
      train_loss, correct, total = 0, 0, 0
      for batch_x, batch_y in train_loader:
          batch_x, batch_y = batch_x.to(device), batch_y.to(device)

          outputs = model(batch_x)
          loss = criterion(outputs, batch_y)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          correct += (outputs.argmax(1) == batch_y).sum().item()
          total += batch_y.size(0)
      train_acc = correct / total
      
      train_acc_l.append(train_acc)
      train_loss_l.append(train_loss)

      model.eval()
      val_loss, val_correct, val_total = 0, 0, 0
      with torch.no_grad():
          for batch_x, batch_y in val_loader:
              batch_x, batch_y = batch_x.to(device), batch_y.to(device)
              outputs = model(batch_x)
              loss = criterion(outputs, batch_y)
              val_loss += loss.item()
              val_correct += (outputs.argmax(1) == batch_y).sum().item()
              val_total += batch_y.size(0)
      val_acc = val_correct / val_total
      
      val_acc_l.append(val_acc)
      val_loss_l.append(val_loss)

      print(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
  
  return train_acc_l, val_acc_l, train_loss_l, val_loss_l


def inference(model, test_loader, device='cpu'):

  """
    Evaluate a trained PyTorch model on a test dataset.

    Input:
        model (nn.Module): Trained PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str or torch.device): Device to run the evaluation on ('cpu' or 'cuda'). Default is 'cpu'.

    Output
        None

    Prints:
        Test accuracy and test loss averaged over the entire test dataset.
    """
    
  model.eval()
  criterion = nn.CrossEntropyLoss()

  curr_test_loss = 0.0
  curr_test_acc = 0
  text_seen_test = 0

  with torch.no_grad():
      for batch_x, batch_y in test_loader:
          
          batch_x, label = batch_x.to(device), batch_y.to(device)
          outputs = model(batch_x)
          test_loss = criterion(outputs, label)

          text_seen_test += len(label)
          curr_test_loss += test_loss.item()
          
          preds = outputs.argmax(1)
          
          curr_test_acc += (preds == label).sum().item()

  avg_test_loss = curr_test_loss / text_seen_test
  avg_test_acc = curr_test_acc / text_seen_test

  print(f"Test Accuracy: {round(avg_test_acc, 4)}, Test Loss: {round(avg_test_loss, 4)}")


def plot_curves(train_loss, train_acc, valid_loss, valid_acc, epochs=5):

  """
    Plot training and validation loss and accuracy curves side by side.

    Input:
        train_loss (list of float): Training loss values per epoch.
        train_acc (list of float): Training accuracy values per epoch.
        valid_loss (list of float): Validation loss values per epoch.
        valid_acc (list of float): Validation accuracy values per epoch.
        epochs (int): Number of epochs. Determines the x-axis range. Default is 5.

    Output:
        None
    """
    
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  axes[0].plot(range(epochs), train_loss, label='Train Loss')
  axes[0].plot(range(epochs), valid_loss, label='Validation Loss')
  axes[0].set_title("Loss Curve")
  axes[0].set_xlabel("Epochs")
  axes[0].set_ylabel("Loss")
  axes[0].legend()

  axes[1].plot(range(epochs), train_acc, label='Train Accuracy')
  axes[1].plot(range(epochs), valid_acc, label='Validation Accuracy')
  axes[1].set_title("Accuracy Curve")
  axes[1].set_xlabel("Epochs")
  axes[1].set_ylabel("Accuracy")
  axes[1].legend()

  plt.tight_layout()
  plt.show()
  
  
def ablation_study(X):
    
    """
    Perform ablation studies on the input feature set by selectively removing or 
    keeping subsets of features to analyze their contribution to model performance.

    Input:
        X (pd.DataFrame): Input dataframe containing features such as demographics, 
                          lab values, vitals, and abnormal flags.

    Output:
        tuple:
            ablation_1 (pd.DataFrame): Dataset with the creatinine abnormal flag 
                                       ('50912_abnormal_flag') removed.
            ablation_2 (pd.DataFrame): Dataset containing only demographic variables 
                                       and raw lab/vital value columns (ending with "_valuenum").

    """
    
    
    ablation_1 = X.drop(['50912_abnormal_flag'], axis=1) # Dropping abnormal flag of creatinine

    demo_cols = ['sex', 'WHITE', 'HISPANIC', 'BLACK/AFRICAN AMERICAN', 'ASIAN', 'OTHER', 'admit_ELECTIVE', 'admit_EMERGENCY', 'admit_URGENT']
    keep_cols = demo_cols + [c for c in X.columns if c.endswith("_valuenum")]
    ablation_2 = X[keep_cols] # Keeping only demographics and raw lab/vital values
    
    return ablation_1, ablation_2
