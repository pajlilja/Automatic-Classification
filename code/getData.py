# -*- coding: utf-8 -*-
"""
@author: Marcus Östling, Joakim Lilja

Get the data from neuromorpho.
"""
from threading import Thread
import sys
import urllib.request, json
#import winsound
import pandas as pd

def getData(species, page, size):
    url = "http://neuromorpho.org/api/neuron/select?q=species:"+species+"&size="+str(size)+"&page="+str(page)
    urldata = urllib.request.urlopen(url)
    data = json.loads(urldata.read().decode())
    return data;
    
def getMorph(neuronDict, neuronDictIndex):
    url = "http://neuromorpho.org/api/morphometry/select?q=neuron_name:"+neuronDict[neuronDictIndex]['neuron_name']
    urldata = urllib.request.urlopen(url)
    data = json.loads(urldata.read().decode())
    try:
        data = data['_embedded']['measurements'][0]
        for key, value in data.items():
            if(key not in neuronDict[neuronDictIndex]):
                neuronDict[neuronDictIndex][key.lower()] = value
    except KeyError:
        print("KeyError:", KeyError, "\nname",neuronDict[neuronDictIndex]['neuron_name'],"\n", data)
            

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Pass a species and .csv file as argument!") 
        sys.exit() 
    
    #winsound.PlaySound('remix.wav',winsound.SND_FILENAME | winsound.SND_ASYNC)
    #Thread(target=f).start()
    
    
    species = sys.argv[1]
    firstData = getData(species,1,500)
    totalPages = firstData['page']['totalPages']
    data = getData(species, 0, 500)
    
    print("Get meta data")
    for page in range(1,totalPages):
        new_data = getData(species, page, 500)
        data['_embedded']['neuronResources'].extend(new_data['_embedded']['neuronResources'])
        print(int((page/totalPages)*100),"%")
        
    data = data['_embedded']['neuronResources']
    
    print("Get morphology")
    lastPercent = ""
    for i in range(0, len(data)):
        getMorph(data,i)
        per = int((i/len(data))*100)
        if per != lastPercent:
            print(per,"%")
            lastPercent = per
        
    df = pd.DataFrame(data)
    df.to_csv(sys.argv[2])
    
    #winsound.SND_PURGE
    #winsound.PlaySound('*',winsound.SND_ALIAS)
