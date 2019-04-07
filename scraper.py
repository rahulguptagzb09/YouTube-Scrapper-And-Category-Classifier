#Import Libraries
from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.tools import argparser

#YouTue Categirues List
cate = {'1': 'Film & Animation', '2': 'Autos & Vehicles', '10': 'Music', '15': 'Pets & Animals', '17': 'Sports', '18': 'Short Movies', '19': 'Travel & Events', '20': 'Gaming', '21': 'Videoblogging', '22': 'People & Blogs', '23': 'Comedy', '24': 'Entertainment', '25': 'News & Politics', '26': 'Howto & Style', '27': 'Education', '28': 'Science & Technology', '30': 'Movies', '31': 'Anime/Animation', '32': 'Action/Adventure', '33': 'Classics', '34': 'Comedy', '35': 'Documentary', '36': 'Drama', '37': 'Family', '38': 'Foreign', '39': 'Horror', '40': 'Sci-Fi/Fantasy', '41': 'Thriller', '42': 'Shorts', '43': 'Shows', '44': 'Trailers'}

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

#YouTube Search List
def youtube_search_list(q, max_results, token=None):
  # Call the search.list method to retrieve results matching the specified
  # query term.
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)

  # Call the search.list method to retrieve results matching the specified
  # query term.
    search_response = youtube.search().list(
        q=q,
        part='id,snippet',
        pageToken=token,
        maxResults=max_results,
        #order='relevance',
        type='video',
        #eventType='completed'
      ).execute()

    return search_response

#Searching Function
def youtube_search_video(q, max_results, token=None):
    max_results = max_results
    #order = "relevance"
    token = token
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)
    q=q

    #Return list of matching records up to max_search
    search_result = youtube_search_list(q, max_results, token)

    #Getting Next Page Token
    try:
        nexttok = search_result["nextPageToken"]
    except Exception as e:
        nexttok = "last_page"

    videos_list = []
    for search_result in search_result.get("items", []):

        if search_result["id"]["kind"] == 'youtube#video':
            temp_dict_ = {}
            #Available from initial search
            temp_dict_['Video id'] = search_result['id']['videoId']
            temp_dict_['Title'] = search_result['snippet']['title']

            #Secondary call to find statistics results for individual video
            response = youtube.videos().list(
                part='statistics, snippet',
                id=search_result['id']['videoId']
                    ).execute()
            response_statistics = response['items'][0]['statistics']
            response_snippet = response['items'][0]['snippet']

            try:
                temp_dict_['Description'] = response_snippet['description']
            except:
                #Not stored if not present
                temp_dict_['Description'] = 'NoneFound'

            try:
                t = response_snippet['categoryId']
                temp_dict_['Category'] = cate[t]
            except:
                #Not stored if not present
                temp_dict_['Category'] = 'NoneFound'

            #add back to main list
            videos_list.append(temp_dict_)

    return (nexttok, videos_list)

#Catgories
q1 = 'Travel'
q2 = 'Science and Technology'
q3 = 'Food'
q4 ='Manufacturing'
q5 = 'History'
q6 = 'Art and Music'
ql = ['Travel', 'Science and Technology', 'Food', 'Manufacturing', 'History', 'Art and Music']

import csv
j = 50;
res = youtube_search_video(q=q4, max_results=50)
token = res[0]
l = res[1]

#Creating CSV File
with open(q4+'_3.csv','w', encoding="utf-8", newline='') as f:
    w = csv.writer(f)
    w.writerow(l[0].keys())
    for i in l:
        w.writerow(i.values())

#Getting either 2000 reuslts or till last page of searched list
while token != "last_page" and j < 2000:
    res = youtube_search_video(q=q4, max_results=50, token=token)
    token = res[0]
    l = res[1]
    j = j+50
    print(j)

    with open(q4+'_3.csv','a', encoding="utf-8", newline='') as f:
        w = csv.writer(f)
        #w.writerow(l[0].keys())
        for i in l:
            w.writerow(i.values())
