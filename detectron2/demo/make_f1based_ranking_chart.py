import os
import heapq
import pickle


def readTextFile(ROOTPATH : str, FILEPATH : str):
    text_result = []
    with open(os.path.join(ROOTPATH, FILEPATH)) as f:
        text_result = f.readlines()
    return text_result


root_PATH = os.path.dirname(os.path.realpath(__file__))
modelnames = ['retinanet_R_50_FPN_1x', 'retinanet_R_50_FPN_3x', 'retinanet_R_101_FPN_3x']

model_result_PATHs = ['retinanet_r_50_fpn_1x_results.txt', 'retinanet_r_50_fpn_3x_results.txt', 'retinanet_r_101_fpn_3x_results.txt']

models_result = []

for FILEPATH in model_result_PATHs:
    models_result.append(readTextFile(root_PATH, FILEPATH))

f1ScoreDict = {}

#first model save
for index in range(2,82):
    datas = models_result[0][index].strip().split()

    if datas[1][0].isnumeric():
        key = datas[0]
        f1Value = datas[3]
        
    else:
        key = ' '.join([datas[0], datas[1]])
        f1Value = datas[4]

    f1ScoreDict[key] = [f1Value]

#second, third model
for modelindex in range(1,len(models_result)):

    for index in range(2,82):
        datas = models_result[modelindex][index].strip().split()

        if datas[1][0].isnumeric():
            key = datas[0]
            f1Value = datas[3]
        else:
            key = ' '.join([datas[0], datas[1]])
            f1Value = datas[4]
            
        f1ScoreDict[key].append(f1Value)

#make ranking per class chart key, index
rankingPerClassChart = {}
for key, value in f1ScoreDict.items():
    ranking = heapq.nlargest(3, range(len(value)), key=value.__getitem__)
    rankingPerClassChart[key] = [(modelnames[rankValue], value[rankValue]) for rankValue in ranking]

# save
with open( os.path.join(root_PATH, f'rankingperclasschart.pickle'), 'wb') as f:
    pickle.dump(rankingPerClassChart, f, pickle.HIGHEST_PROTOCOL)

