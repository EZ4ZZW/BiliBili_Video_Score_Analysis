import numpy as np
import pandas as pd
import os
from urllib.request import Request, urlopen
import urllib
import requests
import json

def getDataFromFile():
    if not os.path.isfile('./data.txt'):
        raise TypeError("file not exist")

    text = open('./data.txt').read()
    return text
    

def getResponse(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.8    5 Safari/537.36'}
    data = requests.get(url = url, headers=headers).text
    return data

url = 'https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all'
res = getDataFromFile()
jsonData = json.loads(res)

title = [] # æ é¢
author= [] # upð·
view  = [] # æ­æ¾é
danmu = [] # å¼¹å¹æ°
coins = [] # æå¸
share = [] # åäº«
like  = [] # ç¹èµ
score = [] # åæ°
favor = [] # æ¶è
reply = [] # è¯è®º
rank  = [] # æå

cnt = 0

for i in range(0, 104):
    temp = ''
    try:
        videoData = jsonData['data']['list'][i]
        score.append(videoData['score'])
        title.append(videoData['title'])
        author.append(videoData['owner']['name'])
        
        videoStat = videoData['stat']
        danmu.append(videoStat['danmaku'])
        view.append(videoStat['view'])
        rank.append(videoStat['his_rank'])
        reply.append(videoStat['reply'])
        favor.append(videoStat['favorite'])
        coins.append(videoStat['coin'])
        share.append(videoStat['share'])
        # test
        like.append(videoStat['like'])
        cnt = cnt + 1
        
    except:
        continue
print(len(title))
print(len(author))
print(len(rank))
print(len(coins))
print(len(score))
print(len(like))
print(len(favor))
print(len(danmu))
print(len(reply))
outfile = pd.DataFrame({"title": title ,"author": author, "rank": rank, "score": score, "view": view, "coins": coins, "favorite": favor, "reply": reply, "danmu": danmu})
outfile.to_csv("bilibili2.csv", index = False)
