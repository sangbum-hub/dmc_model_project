import pandas as pd 
from settings import *

info_list = glob.glob(f'{DATA_DIR}/info/*.csv')

df_2021_pre = pd.read_csv(info_list[0])
df_2021_sta = pd.read_csv(info_list[1])
df_2022_pre = pd.read_csv(info_list[2])
df_2022_sta = pd.read_csv(info_list[3])


rename_manu = {
    '사업자등록번호': 'CompanyNo', 
    '기업명':'CompanyName', 
    '고객 세그먼트': 'CustomerSeg', 
    'label': 'Seg_label',
    '가치제안':'ValueProposition', 
    'label.1': 'Value_label',
    '마케팅채널':'Channels', 
    'label.2':'Ch_label',
    '고객관계':'CustomerRelation', 
    'label.3':'Relation_label',
    '수익원':'Revenue', 
    'label.4':'Rev_label',
    '핵심활동':'KeyActivities',
    'label.5':'Act_label', 
    '핵심자원':'KeyResources',
    'label.6':'Res_label', 
    '핵심파트너':'KeyPartners',
    'label.7': 'Part_label', 
    '비용구조':'Cost',
    'label.8':'Cost_label'
}




target_col = [
    'CompanyNo', 'CompanyName', 'CustomerSeg', 'ValueProposition', 
    'Channels', 'CustomerRelation', 'Revenue', 'KeyActivities', 
    'KeyResources', 'KeyPartners', 'Cost']


df_2021_pre = df_2021_pre.rename(columns=rename_manu)
df_2021_sta = df_2021_sta.rename(columns=rename_manu)
df_2022_pre = df_2022_pre.rename(columns=rename_manu)
df_2022_sta = df_2022_sta.rename(columns=rename_manu)


values = ['CustomerSeg', 'ValueProposition', 'Channels', 'CustomerRelation', 'Revenue', 'KeyActivities', 'KeyResources', 'KeyPartners', 'Cost']
labels = ['Seg_label', 'Value_label', 'Ch_label', 'Relation_label', 'Rev_label', 'Act_label', 'Res_label', 'Part_label', 'Cost_label']

for value, label in zip(values, labels):
    cols = ['CompanyNo', 'CompanyName'] + [value, label]
    total = pd.concat([df_2021_pre.loc[:, cols], df_2021_sta.loc[:, cols], df_2022_pre.loc[:, cols], df_2022_sta.loc[:, cols]], axis=0)
    total.rename(columns={label:'label'}, inplace=True)
    total.to_csv(f'{DATA_DIR}/{value}.csv', encoding='utf-8-sig', index=False)
