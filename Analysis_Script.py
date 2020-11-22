#!/usr/local/bin/python3
# File created on 2020/09/02
# File most recently updated on 2020/09/14 (added functions from fuzzywuzzy) as well as optmizes argument parsing and time efficiency
# File most recently updated on 2020/09/17 added codes to check latest version of df_files are saved


# Basic Info
__author__ = "Zhang, Jianxiang (Tom)"
__copyright__ = "Copyright 2020, Zhang, Jianxiang (Tom), UC Berkeley"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "tom_zhang_2023@berkeley.edu"

# Loading modules
from utils import *
import argparse
import pandas as pd
import pyreadr
import warnings
from scipy.optimize import OptimizeWarning
import os
from datetime import date, datetime
import shutil

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=OptimizeWarning)

pd.options.mode.chained_assignment = None

def options():
    parser = argparse.ArgumentParser(description = "This is a python script for analyzing COVID-19 Cases. It includes two main functions: \n\
                                                    1. fitting curves and forecasting future cases\n\
                                                    2. mapping data on maps and show visualize the progress of this pandemic.\n\
                                                    Before running this script, you need to run the shell script to download the latest data from the JHU repo \n\
                                                    and run R file to perform preliminary cleaning and extraction of these data.\n\
                                                    The necessary modules for the two functions are: 'pandas', 'numpy', 'matplotlib', \n\
                                                    're', 'scipy', 'mpl_tookits', 'plotly', 'pycountry'.\n\
                                                    If you'd like, you can install 'fuzzywuzzy' to allow for errors in your input, but it is not required. \n\
                                                    Regional mapping are currently available for U.S. and China.\n\
                                                    P.S.: Please change all blanks in parameter name into underlines to avoid parsing errors. \n\
                                                    More functions will hopefully be on the way. Have Fun!") 

    parser.add_argument("--df", action = "store", dest = "df", required= True,
                        help = "name of dataframe or csv file path; \n\
                            Available names without supplying a csv file are: \n\
                            'global_counts_total', 'us_counts_total', 'china_counts_total', 'global_counts_new', 'us_counts_new', 'china_counts_new', \n\
                            'global_counts_total_geo', 'us_counts_total_geo', 'china_counts_total_geo', 'global_counts_new_geo', 'us_counts_new_geo', 'china_counts_new_geo', \n\
                            'global_counts_origin_total', 'global_counts_origin_new'", 
                            default = False)
    parser.add_argument("--name", action = "store", dest = "name",  
                        help = "name of region for analysis", default = "") 
    parser.add_argument("--forecast", action = "store", dest = "forecast",  
                        help = "forecast (True) or don't forecast (False)", default = False)
    parser.add_argument("--arima", action = "store", dest = "arima",  
                        help = "use arima for forecast (True) or don't use arima (False)", default = False)
    parser.add_argument("--scale", action = "store", dest = "scale",  
                        help = "scale when plotting map gifs: 'continuous' (default) or 'log'", default = 'continuous') 
    parser.add_argument("--save", action = "store", dest = "save",  
                        help = "provide a directory for saving forecast plot/map (but not both) or don't save (enter False)", default = False) 
    parser.add_argument("--show", action = "store", dest = "show",  
                        help = "show forecast plot AND map plot (True) or don't show (False)", default = False)
    parser.add_argument("--map", action = "store", dest = "map",  
                        help = "plot map (True) or don't plot (False)", default = False)
    parser.add_argument("--json", action = "store", dest = "json",  
                        help = "provide a geojson file corresponding to your df argument or enter False", default = False)
    parser.add_argument("--print", action = "store", dest = "print",  
                        help = "print given df file (True) or don't print (False)", default = False)

                    
    # Read arguments 
    args = vars(parser.parse_args())
    if "name" in args.keys():
        args["name"] = args["name"].replace("_", " ")
    
    # Return argument values 
    return args

def get_data():
    global_counts_total = pd.read_csv('df_files/global_counts_total.csv')
    us_counts_total = pd.read_csv('df_files/us_counts_total.csv')
    china_counts_total = pd.read_csv('df_files/china_counts_total.csv')
    global_counts_new = pd.read_csv('df_files/global_counts_new.csv')
    us_counts_new = pd.read_csv('df_files/us_counts_new.csv')
    china_counts_new = pd.read_csv('df_files/china_counts_new.csv')
    global_counts_total_geo = pd.read_csv('df_files/global_counts_total_geo.csv')
    us_counts_total_geo = pd.read_csv('df_files/us_counts_total_geo.csv')
    china_counts_total_geo = pd.read_csv('df_files/china_counts_total_geo.csv')
    global_counts_new_geo = pd.read_csv('df_files/global_counts_new_geo.csv')
    us_counts_new_geo = pd.read_csv('df_files/us_counts_new_geo.csv')
    china_counts_new_geo = pd.read_csv('df_files/china_counts_new_geo.csv')

    global_counts_origin_total = pd.read_csv('jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                                  header = 0)
    global_counts_origin_new_dict = {}
    for i in range(0, len(global_counts_origin_total.columns)):
        if i < 4:
            global_counts_origin_new_dict[global_counts_origin_total.columns[i]] = global_counts_origin_total[global_counts_origin_total.columns[i]]
        else:
            global_counts_origin_new_dict[global_counts_origin_total.columns[i]] = global_counts_origin_total[global_counts_origin_total.columns[i]] - global_counts_origin_total[global_counts_origin_total.columns[i-1]]
    global_counts_origin_new = pd.DataFrame(global_counts_origin_new_dict)

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
    return {'w/o geo': data_dict, 'with geo':data_dict_geo}

def retrieve_df():
    '''
    retrieve dfs from existing csv files from previous runs
    '''
    data_dict = {}
    data_dict_geo = {}
    # reference: https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
    for filename in os.listdir('df_files'):
        if filename.endswith("geo.csv"):
            data_dict_geo[filename[:-4]] = pd.read_csv('df_files/'+filename, index_col = 0)
        else:
            data_dict[filename[:-4]] = pd.read_csv('df_files/'+filename, index_col = 0)
    return {'w/o geo': data_dict, 'with geo':data_dict_geo}


def closest_match(given_name, lst):
    if given_name == False:
        print(">>> ERROR: Please provide a valid name from: ", lst)
        exit(0)
    elif given_name not in lst:
        try:
            from fuzzywuzzy import fuzz
            from fuzzywuzzy import process
            print('>>> Attempt to match your input name —— ' + given_name + ' —— with existing names')
            closest = process.extractOne(given_name, lst)
            print('>>> Match name is: ', closest[0], ' Match score is: ', closest[1])
            if closest[1] >= 80:
                return closest[0]
            else:
                print(">>> ERROR: Please provide a valid name from: ", lst)
                exit(0)
        except ModuleNotFoundError as e:
            print(">>> ERROR: Please provide a valid name from: ", lst)
            exit(0)

def variable_process(args):
    params = {}
    if ".csv" in args["df"]:
        is_csv = True
        print(">>> WARNING: unexpected behavior might appear if input file is a raw csv.")
        if "total" not in args["df"] and "new" not in args["df"]:
            print(">>> ERROR: Please include 'total' or 'new' in your filename to indicate whether it is a datafile of accumulated cases up to each date or daily new cases.")
            print(">>> Exit...")
            exit(0)
        elif "global" not in args["df"] and "us" not in args["df"] and "US" not in args["df"]:
            print(">>> ERROR: Please include 'global' or 'us' in your filename to indicate whether it is a datafile of global cases or us cases.")
            print(">>> Exit...")
            exit(0)
        else:
            if "global" in args["df"]:
                region = "global"
            elif "us" in args["df"]:
                region = "us"
            else:
                region = "name"
            df = pd.read_csv(args["df"], header = 0)
            df = df_tidy(df, region)
            df_geo = remove_rows(df)
            df = remove_geo(df)
            params["df_geo"] = df_geo
    else:
        modified_time = datetime.fromtimestamp(os.path.getmtime('df_files')).strftime('%Y-%m-%d')
        modified_time = datetime.strptime(modified_time, '%Y-%m-%d').date()
        if os.path.exists('df_files') and modified_time >= date.today(): # do not waste time generates dfs once more
            data = retrieve_df()
        else:
            data = get_data()
        data_dict = data['w/o geo']
        data_dict_geo = data['with geo']
        is_csv = False
        name = args["name"]
        if args["df"] not in data_dict.keys() and args["df"] not in data_dict_geo.keys():
            name_df = closest_match(args["df"], list(data_dict.keys())+list(data_dict_geo.keys()))
            try:
                df = data_dict[name_df]
            except:
                df = data_dict_geo[name_df]
        elif args["df"] == "global_counts_origin_total" or args["df"] == "global_counts_origin_new":
            if "total" in args["df"]:
                global_counts_origin = data_dict["global_counts_origin_total"]
            else:
                global_counts_origin = data_dict["global_counts_origin_new"]
            lst_countries = global_counts_origin["Country/Region"].tolist()
            name = closest_match(name, lst_countries)
            country_counts = global_counts_origin.loc[global_counts_origin_total['Country/Region'] == name]
            country_counts = country_counts[[country_counts.columns[0]]+list(country_counts.columns[2:])]
            country_counts.rename(columns = {'Province/State': 'name', 'Lat': 'lat', 'Long': 'lon'}, inplace = True)
            df = country_counts
            region = "country"
        else:
            if "geo" in args["df"]:
                df = data_dict_geo[args["df"]]
            else:
                df = data_dict[args["df"]]
        if "global" in args["df"] and "origin" not in args["df"]:
            region = "global"
        elif "us" in args["df"]:
            region = "us"
        elif "china" in args["df"]:
            region = "china"
    if "total" in args["df"]:
        choice = "total"
    else:
        choice = "new"
    params.update({'data_dict': data_dict, 'data_dict_geo': data_dict_geo, 
                    'csv?': is_csv, 'df_name': args["df"], 'df': df, 
                    'region': region, 'choice': choice, 'name': name, 
                    'scale': args["scale"], 'save': args["save"], 'show': args["show"], 
                    'print': args["print"], "arima": args["arima"]})
    if args["json"] != False:
        params["json"] = args["json"]
    return params

def forecast_analysis(params):
    data_dict = params["data_dict"]
    if not params["csv?"]:
        if "geo" in params["df_name"]:
            print(">>> ERROR: Wrong df file. Must be one of " + str([key for key in data_dict.keys()]))
            print(">>> Exit...")
            exit(0)
    if params["save"] != False and params["save"] != "False":
        save = params["save"]
    else:
        save = "."
    if params["arima"] == "True":
        try:
            df_new = forecast_arima(params["df"], params["name"], params["choice"], show = params["show"], save = save)
        except KeyError as e:
            name = params["name"]
            name = closest_match(name, params["df"].index.tolist())
            df_new = forecast_arima(params["df"], name, params["choice"], show = params["show"], save = save)
        print(df_new)
    else:
        try:
            df_new = forecast(params["df"], params["region"], params["name"], params["choice"], show = params["show"], save = save)
        except KeyError as e:
            name = params["name"]
            name = closest_match(name, params["df"].index.tolist())
            df_new = forecast(params["df"], params["region"], name, params["choice"], show = params["show"], save = save)
        print(df_new)

def plot_map(params):
    data_dict_geo = params['data_dict_geo']
    if not params["csv?"]:
        if "geo" not in params["df_name"] and "origin" not in params["df_name"]:
            print(">>> ERROR: Wrong df file. Must be one of " + str([key for key in data_dict_geo.keys()]))
            print(">>> Exit...")
            exit(0)
        df = params["df"]
    else:
        df = params["df_geo"]
    if "json" in list(params.keys()):
        print(">>> Retreiveing json file...")
        jsonfile = params["json"]
        plot_alt(df, params["region"], params["choice"], params["scale"], params["save"], params["show"], jsonfile)
    else:
        plot_gif(df, params["region"], params["choice"], params["scale"], params["save"], params["show"])


# Define a main function to call other functions 
def main():
    '''
        The main function
    '''
    # Get arguments from command lines 
    args = options()
    print(">>> Inputs arguments are: ", args)
    if len(args) < 3:
        print('>>> ERROR: Please input necessary argument values!')
        exit(0)

    # Print start info 
    print (">>> Start...")
    
    # Call function 
    params = variable_process(args)
    if args["print"] == "True":
        print(params["df"])
    if args["forecast"] == "True":
        print('>>> Starting Forecast Analysis...')
        forecast_analysis(params)
    if args["map"] == "True":
        print('>>> Plotting Maps...')
        plot_map(params)
    
    # Print end info 
    print (">>> Finished...")
    

# Execute functions 
if __name__ == '__main__':
    main()