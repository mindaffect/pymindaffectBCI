def get_thresholds(file):    
    import json
    from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_data_messages
    from mindaffectBCI.utopiaclient import Log
    from collections import defaultdict

    if isinstance(file,str):
        _, messages = read_mindaffectBCI_data_messages(file)
    else:
        messages = file
    log_messages = [m.logmsg for m in messages if m.msgID==Log.msgID and 'detection_threshold' in m.logmsg]
    if len(log_messages) == 0:
        return None
    list_thresholds = [json.loads(log) for log in log_messages]
    organized_logs = defaultdict(list)

    for log in list_thresholds: #transform the list into a dict
        for key,value in log.items():
            organized_logs[key].append(value)
    return organized_logs['stimulus'] #return the threshold

def get_user_response(file):
    import json
    from mindaffectBCI.decoder.offline.read_mindaffectBCI import read_mindaffectBCI_data_messages
    from mindaffectBCI.utopiaclient import Log
    import pandas as pd
    from collections import defaultdict

    if isinstance(file,str):
        _, messages = read_mindaffectBCI_data_messages(file)
    else:
        messages = file
    log_messages = [m.logmsg for m in messages if m.msgID==Log.msgID and 'user_response' in m.logmsg]
    #organize the messages, load them properly
    list_user_r = [json.loads(log) for log in log_messages]
    organized_logs = defaultdict(list)

    for log in list_user_r: 
        for key,value in log.items():
            organized_logs[key].append(value)
    df_user_r = pd.DataFrame.from_dict(organized_logs) #transform the dict into dataframe
    return df_user_r

def plot_staircase(user_r):
    import matplotlib.pyplot as plt

    all_lvl_i = []
    for x in range(len(user_r)):
        i = user_r['stimulus'][x]
        all_lvl_i.append(i['level_idx'])

    #get trial number for thresholds for the vertical lines
    i_thr = user_r[user_r['stimulus'] == {'level_idx': 7, 'level': 0.003162}].index

    plt.plot(all_lvl_i, 'black')
    plt.xlabel('Trials')
    plt.ylabel('Volume level')
    for x in range(1,len(i_thr)): #the first few lines
        plt.axvline(i_thr[x]-1, color = 'r', linestyle = '-', linewidth=1)
    plt.axvline(len(user_r)-1, color = 'r', linestyle = '-', linewidth=1) #the last line
    plt.grid(axis='y')
