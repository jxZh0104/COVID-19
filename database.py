import pandas as pd
import pyreadr
import sqlite3
import os
from datetime import datetime
from datetime import date
from dateutil.parser import parse
from utils import *
import shutil
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 

pd.options.mode.chained_assignment = None

def get_data():
    try:
        result = pyreadr.read_r("COVID-19_cleaned_cases.RData")
    except:
        print(">>> ERROR: Some Error has occured. It is possible that you have not run the Rscript provided to gather preliminary RData file.")
        print(">>> Exit...")
        exit(0)

    global_counts_total = df_tidy(result['total_counts_global'], "global")
    us_counts_total = df_tidy(result['total_counts_us'], "us")
    global_counts_new = df_tidy(result['new_counts_global'], "global")
    us_counts_new = df_tidy(result['new_counts_us'], "us")
    global_counts_origin_total = pd.read_csv('jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                                  header = 0)
    global_counts_origin_new_dict = {}
    for i in range(0, len(global_counts_origin_total.columns)):
        if i < 4:
            global_counts_origin_new_dict[global_counts_origin_total.columns[i]] = global_counts_origin_total[global_counts_origin_total.columns[i]]
        else:
            global_counts_origin_new_dict[global_counts_origin_total.columns[i]] = global_counts_origin_total[global_counts_origin_total.columns[i]] - global_counts_origin_total[global_counts_origin_total.columns[i-1]]
    global_counts_origin_new = pd.DataFrame(global_counts_origin_new_dict)

    # generate counts_df for china
    china_counts = global_counts_origin_total.loc[global_counts_origin_total['Country/Region'] == 'China']
    addition = global_counts_origin_total.loc[global_counts_origin_total['Country/Region'] == 'Taiwan*'] 
    addition['Province/State'] = 'Taiwan'
    addition['Country/Region'] = 'China'
    china_counts = pd.concat([china_counts, addition])
    china_counts_total = china_counts[[china_counts.columns[0]]+list(china_counts.columns[2:])]
    china_counts_total.rename(columns = {'Province/State': 'province', 'Lat': 'lat', 'Long': 'lon'}, inplace = True)
    china_dict = {}
    for i in range(len(china_counts_total.index)):
        china_dict[china_counts_total.index[i]] = list(china_counts_total['province'])[i]
    china_counts_total.rename(index = china_dict, inplace = True)

    china_counts_new_dict = {}
    for i in range(0, len(china_counts_total.columns)):
        if i < 4:
            china_counts_new_dict[china_counts_total.columns[i]] = china_counts_total[china_counts_total.columns[i]]
        else:
            china_counts_new_dict[china_counts_total.columns[i]] = china_counts_total[china_counts_total.columns[i]] - china_counts_total[china_counts_total.columns[i-1]]
    china_counts_new = pd.DataFrame(china_counts_new_dict, index = list(china_counts_total['province']))
    global_counts_total_geo = remove_rows(global_counts_total)
    us_counts_total_geo = remove_rows(us_counts_total)
    china_counts_total_geo = remove_rows(china_counts_total)
    global_counts_new_geo = remove_rows(global_counts_new)
    us_counts_new_geo = remove_rows(us_counts_new)
    china_counts_new_geo = remove_rows(china_counts_new)

    global_counts_total = remove_geo(global_counts_total)
    us_counts_total = remove_geo(us_counts_total)
    china_counts_total = remove_geo(china_counts_total)
    global_counts_new = remove_geo(global_counts_new)
    us_counts_new = remove_geo(us_counts_new)
    china_counts_new = remove_geo(china_counts_new)

    data_dict = {'global_counts_total': global_counts_total,
            'us_counts_total': us_counts_total,
            'china_counts_total': china_counts_total,
            'global_counts_new': global_counts_new,
            'us_counts_new': us_counts_new,
            'china_counts_new': china_counts_new,
            'global_counts_origin_total': global_counts_origin_total,
            'global_counts_origin_new': global_counts_origin_new
            }
    data_dict_geo = {'global_counts_total_geo': global_counts_total_geo,
            'us_counts_total_geo': us_counts_total_geo,
            'china_counts_total_geo': china_counts_total_geo,
            'global_counts_new_geo': global_counts_new_geo,
            'us_counts_new_geo': us_counts_new_geo,
            'china_counts_new_geo': china_counts_new_geo
            }

    # output txt files with the rownames of each of these columns:
    if not os.path.exists('data_entries'):
        os.makedirs('data_entries')
        for name in list(data_dict_geo.keys()):
            with open('data_entries/' + name[:-4] + '.txt', 'w+') as f:
                for region in list(data_dict_geo[name].index):
                    f.write(region + '\n')
    # output csv files into another dorectory:
    if os.path.exists('df_files'):
        shutil.rmtree('df_files')
    os.makedirs('df_files')
    for name in list(data_dict.keys()):
        data_dict[name].to_csv('df_files/' + name + '.csv')
    for name in list(data_dict_geo.keys()):
        data_dict_geo[name].to_csv('df_files/' + name + '.csv')
    
    # download dataset for Berkeley
    data = pd.read_csv('https://data.cityofberkeley.info/resource/xn6j-b766.csv')
    data.iloc[:,0] = [date[:10] for date in data.iloc[:,0]]
    newData = pd.DataFrame(data = {"Date": data.iloc[:,0], 
                                    "Daily Cases": data.iloc[:,1],
                                    "Accumulative Cases": data.iloc[:,2]})
    newData.to_csv('df_files/Berkeley.csv')

# check if the dataset is up-to-date
modified_time = datetime.fromtimestamp(os.path.getmtime('df_files')).strftime('%Y-%m-%d')
modified_time = datetime.strptime(modified_time, '%Y-%m-%d').date()
if not (os.path.exists('df_files') and modified_time >= date.today()): # do not waste time generates dfs once more
    get_data()

geoFiles = []
tableNames = []
for dirname, _, filenames in os.walk('df_files'):
    for filename in filenames:
        if filename[-7:-4] == 'geo':
            geoFiles.append(os.path.join(dirname, filename))
            tableNames.append(filename[:-8])

def createTable(cursorObj, fileName, tableName):
    cursorObj.execute("DROP TABLE IF EXISTS " + tableName)
    data = pd.read_csv(fileName, index_col = 0)
    columns = data.columns
    dateColumns = []
    for c in columns[3:]:
        dt = parse(c)
        date = dt.date()
        dateColumns.append("Date" + date.strftime("%m%d%y"))
    createDateColumnCommand = " INTEGER, ".join(dateColumns) + " INTEGER);"
    createTableCommand = "CREATE TABLE " + tableName + " (" + data.columns[0] + " TEXT PRIMARY KEY, lat NUMERIC, lon NUMERIC, " + createDateColumnCommand
    cursorObj.execute(createTableCommand)
    insertDateCommand = ", ".join(dateColumns) + ") "
    insertTableCommand = "INSERT INTO " + tableName + " (" + data.columns[0] + ", lat, lon, " + insertDateCommand + "VALUES (" + "?, " * (len(data.columns) - 1) + "?);"
    to_db = [tuple(data.iloc[i, :]) for i in range(len(data))]
    cursorObj.executemany(insertTableCommand, to_db)

con = sqlite3.connect("sqlite3/COVID_database.db")
cursorObj = con.cursor()
for i in range(len(geoFiles)):
    createTable(cursorObj, geoFiles[i], tableNames[i])

# create database for Berkeley
data = pd.read_csv('df_files/Berkeley.csv', index_col = 0)
cursorObj.execute("DROP TABLE IF EXISTS Berkeley")
cursorObj.execute("CREATE TABLE Berkeley(Date TEXT PRIMARY KEY, New INTEGER, Total INTEGER);")
to_db = [tuple([data.iloc[i, 0], int(data.iloc[i, 1]), int(data.iloc[i, 2])]) for i in range(len(data))]
cursorObj.executemany("INSERT INTO Berkeley(Date, New, Total) VALUES (?, ?, ?);", to_db)
#cursorObj.execute("SELECT * FROM Berkeley")
#print(cursorObj.fetchall())


con.commit()
con.close()







