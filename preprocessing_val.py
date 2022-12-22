import pandas as pd 
from settings import *

info_list = glob.glob(f'{DATA_DIR}/info/*.csv')

df_2021_pre = pd.read_csv(info_list[0])
df_2021_sta = pd.read_csv(info_list[1])
df_2022_pre = pd.read_csv(info_list[2])
df_2022_sta = pd.read_csv(info_list[3])
# it = pd.read_csv(info_list[1], encoding='cp949')


rename_manu = {
    '번호': 'CompanyNo', 
    '기업명':'CompanyName', 
    '창업동기 및 목표의 구체성': 'chmok', 
    'label': 'chmok_label',
    '창업 분야 이해도(사업 도메인, 시장)':'che', 
    'label.1': 'che_label',
    '창업자(팀원)의 전문성':'chj', 
    'label.2':'chj_label',
    '창업 아이템(기능, 특징 등의 우수성)':'chi', 
    'label.3':'chi_label',
    '경쟁 제품과의 차별성':'pdd', 
    'label.4':'pdd_label',
    '창업 아이템의 시장성':'ps',
    'label.5':'ps_label', 
    '사업 계획서 내용의 충실성':'sch',
    'label.6':'sch_label', 
    '시장 진출 및 성장 계획의 실현 가능성':'sss',
    'label.7': 'sss_label'
}


target_col = [
    'CompanyNo', 'CompanyName', 'chmok', 'che', 
    'chj', 'chi', 'pdd', 'ps', 
    'sch', 'sss']


df_2021_pre = df_2021_pre.rename(columns=rename_manu)
df_2021_sta = df_2021_sta.rename(columns=rename_manu)
df_2022_pre = df_2022_pre.rename(columns=rename_manu)
df_2022_sta = df_2022_sta.rename(columns=rename_manu)
# re_it = it.rename(columns=rename_it)

values = ['chmok', 'che', 'chj', 'chi', 'pdd', 'ps', 'sch', 'sss']
labels = ['chmok_label', 'che_label', 'chj_label', 'chi_label', 'pdd_label', 'ps_label', 'sch_label', 'sss_label']

for value, label in zip(values, labels):
    cols = ['CompanyNo', 'CompanyName'] + [value, label]
    total = pd.concat([df_2021_pre.loc[:, cols], df_2021_sta.loc[:, cols], df_2022_pre.loc[:, cols], df_2022_sta.loc[:, cols]], axis=0)
    # total = pd.DataFrame(df_2021_pre.loc[:, cols])
    total.rename(columns={label:'label'}, inplace=True)
    total.to_csv(f'{DATA_DIR}/{value}.csv', encoding='utf-8-sig', index=False)
