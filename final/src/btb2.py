import time; start_time = time.time()
import numpy as np
import pandas as pd
import subprocess
import pickle
#BTB 0.63523 + evaluation
#by clustifier
print('Python Script...')
df_test = pd.read_csv('../input/clicks_test.csv')
df_ad = pd.read_csv('../input/promoted_content.csv', usecols  = ('ad_id','document_id'), dtype={'ad_id': np.int, 'uuid': np.str, 'document_id': np.str})
df_events = pd.read_csv('../input/events.csv', usecols  = ('display_id','uuid','timestamp'), dtype={'display_id': np.int, 'uuid': np.str, 'timestamp': np.int})
df_test = pd.merge(df_test, df_ad, on='ad_id', how='left')
df_test = pd.merge(df_test, df_events, on='display_id', how='left')
df_test['usr_doc'] = df_test['uuid'] + '_' + df_test['document_id']
df_test = df_test.set_index('usr_doc')
time_dict = df_test[['timestamp']].to_dict()['timestamp']
f = open("../input/page_views.csv", "r")
line = f.readline().strip()
head_arr = line.split(",")
fld_index = dict(zip(head_arr,range(0,len(head_arr))))
total = 0
found = 0
while 1:
    line = f.readline().strip()
    total += 1
    if total % 100000000 == 0:
        print('Read {} lines, found {}'.format(total,found))
    if line == '':
        break
    arr = line.split(",")
    usr_doc = arr[fld_index['uuid']] + '_' + arr[fld_index['document_id']]
    if usr_doc in time_dict:
            time_dict[usr_doc] = -1
            found += 1
print(found)
df_test=df_test.reset_index()
df_test['fixed_timestamp'] = df_test['usr_doc'].apply(lambda x: time_dict[x])
train = pd.read_csv("../input/clicks_train.csv")
cnt = train[train.clicked==1].ad_id.value_counts()
cntall = train.ad_id.value_counts()
ave_ctr = np.sum(cnt)/float(np.sum(cntall))
def get_prob(x):
    if x[0] < 0:
        return 1
    k = x[1]
    if k in cnt:
        return cnt[k] / (float(cntall[k]) + 10)
    else:
        if k in cntall:
            return -1 * cntall[k]
        else:
            return ave_ctr
def agg2arr(x):
    return list(x)
def val_sort(x):
    id_dict = dict(zip(x[0], x[1]))
    id_list_sorted =  [k for k,v in sorted(id_dict.items(), key=lambda x:x[1], reverse=True)]
    return " ".join(map(str,id_list_sorted))
df_test['prob'] = df_test[['fixed_timestamp','ad_id']].apply(lambda x: get_prob(x),axis=1)
subm = df_test.groupby("display_id").agg({'ad_id': agg2arr, 'prob': agg2arr})
subm['ad_id'] = subm[['ad_id','prob']].apply(lambda x: val_sort(x),axis=1)
del subm['prob']
subm.to_csv("SubmissionP.csv")