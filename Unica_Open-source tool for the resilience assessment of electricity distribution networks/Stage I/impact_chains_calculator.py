
"""
@author: Adriano Casu, University of Cagliari, e-mail: adriano.casu@unica.it

"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer


def upload_indicators():
    
    while True:
        try:
            file_path = 'indicators.xlsx'
            df_indicators = pd.read_excel(file_path)
            return df_indicators
        except FileNotFoundError:
            print(f"FileNotFoundError: '{file_path}' does not exist. Try again...")

def upload_weights():
    
    while True:
        try:
            file_path = 'weights.xlsx'
            df_weights = pd.read_excel(file_path)
            return df_weights
        except FileNotFoundError:
            print(f"FileNotFoundError: '{file_path}' does not exist. Try again...")

def normalization(df_indicators):
    
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    df_log = pd.DataFrame(log_transformer.fit_transform(df_indicators), columns=df_indicators.columns)
    
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_log), columns=df_log.columns)
    df_norm['ID'] = df_indicators['ID']
    
    return df_norm

def compute_indices(event):
    
    df_indicators = upload_indicators()
    df_weights = upload_weights()
    df_norm = normalization(df_indicators)
    chain_weights = {'hazard': 1, 'exposure': 3, 'sensitivity': 1, 'adaptation': 1}
    
    hazard = df_norm.climate_index
    
    exposure = (df_norm.L_underground_cables * df_weights.L_underground_cables.iloc[0] + df_norm.N_joints * df_weights.N_joints.iloc[0] + df_norm.N_terminals * df_weights.N_terminals.iloc[0] + df_norm.joints_avg_age * df_weights.joints_avg_age.iloc[0] + df_norm.L_overhead_lines * df_weights.L_overhead_lines.iloc[0] + df_norm.conductors_avg_age * df_weights.conductors_avg_age.iloc[0] + df_norm.poles_material * df_weights.poles_material.iloc[0])/(df_weights.L_underground_cables.iloc[0] + df_weights.N_joints.iloc[0] + df_weights.N_terminals.iloc[0] + df_weights.joints_avg_age.iloc[0] + df_weights.L_overhead_lines.iloc[0] + df_weights.conductors_avg_age.iloc[0] + df_weights.poles_material.iloc[0])
    if event == 'HW':
        grid_check = [c for c in df_indicators.columns if c in ['L_overhead_lines', 'L_underground_cables', 'N_joints', 'N_terminals']]
        exposure[df_indicators[grid_check].sum(axis=1) == 0] = 0
    elif event == 'WS':
        grid_check = [c for c in df_indicators.columns if c in ['L_overhead_lines']]
        exposure[df_indicators[grid_check].sum(axis=1) == 0] = 0
    
    sensitivity = (df_norm.LV_users * df_weights.LV_users.iloc[0] + df_norm.MV_users * df_weights.MV_users.iloc[0] + df_norm.flexible_users * df_weights.flexible_users.iloc[0] + df_norm.premium_users * df_weights.premium_users.iloc[0])/(df_weights.LV_users.iloc[0] + df_weights.MV_users.iloc[0] + df_weights.flexible_users.iloc[0] + df_weights.premium_users.iloc[0])
    
    adaptation = 1 - (df_norm.installed_power * df_weights.installed_power.iloc[0] + df_norm.trunk_nodes * df_weights.trunk_nodes.iloc[0])/(df_weights.installed_power.iloc[0] + df_weights.trunk_nodes.iloc[0])
    
    risk = (hazard * chain_weights['hazard'] + exposure * chain_weights['exposure'] + sensitivity * chain_weights['sensitivity'] + adaptation * chain_weights['adaptation'])/sum(chain_weights.values())
    risk[(hazard == 0) | (exposure== 0)] = 0
    
    list_df = [df_indicators.ID, hazard, exposure, sensitivity, adaptation, risk]
    header = ['ID', 'Hazard', 'Exposure', 'Sensitivity', 'Adaptation', 'Risk']
    df_results = pd.concat(list_df, axis=1, keys=header)
    
    return df_results

def create_output(df_results):
    
    file_path = 'results.xlsx'
    df_results.to_excel(file_path, sheet_name = 'Results',index=False)

event = 'HW' # choose between 'HW' (heatwaves) or 'WS' (windstorms)
results = compute_indices(event)
create_output(results)

