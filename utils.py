import pyreadr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
from scipy import optimize
from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML, Image
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import json
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose


# update: forecast_arima() # 09/15/2020
def forecast_arima(dataset, name, choice, figsize = (20, 5), pred_ahead = 10, freq = None, show = False, save = "."):
    '''
    dataset can only be data of new cases per month, i.e. choice = 'new'
    '''
    if choice != "new":
        print('>>> ERROR: Please feed dataset on new daily cases for arima analysis.')
        exit(0)
    data = pd.DataFrame(data = {'value': dataset.loc[name].tolist()}, 
                   index = pd.to_datetime(dataset.columns))
    smodel = pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3,m = 7, # m=7 corresponds to daily data
                         start_P=0, seasonal=True,
                         d=1, D=1, trace=False,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
    train = data.loc['2020-01-22':'2020-09-01']
    test = data.loc['2020-09-01':]
    smodel.fit(train)
    future_forecast, confint = smodel.predict(n_periods=len(test)+pred_ahead, return_conf_int = True)
    test_date = pd.date_range(start=test.index[0],periods=len(test)+pred_ahead, freq = None)
    future_forecast = pd.DataFrame(future_forecast,index = test_date,columns=['Prediction'])
    lower_series = pd.Series(confint[:, 0], index=test_date)
    upper_series = pd.Series(confint[:, 1], index=test_date)
    lower_series = pd.DataFrame(data = {'lower':confint[:, 0]}, index=test_date)
    upper_series = pd.DataFrame(data = {'upper':confint[:, 1]}, index=test_date)
    predictions = pd.DataFrame(data = {'actual': test['value'].tolist() + [np.nan]*pred_ahead, 
                                       'lower': lower_series['lower'].tolist(), 
                                       'prediction': future_forecast['Prediction'].tolist(), 
                                       'upper':upper_series['upper'].tolist()},
                                index = test_date)
    plt.figure(figsize = figsize)
    plt.plot(data, label = 'Actual')
    plt.plot(future_forecast, label = 'Prediction')
    plt.fill_between(lower_series.index, 
                 lower_series['lower'],
                 upper_series['upper'], 
                 color='k', alpha=.15, label = 'Uncertainty')
    plt.title('Forecast Plot of COVID-19 Cases in ' + name.title(), fontsize = 20)
    plt.xlabel('Date', fontsize = 15)
    plt.ylabel('Daily Confirmed Cases', fontsize = 15)
    plt.legend() # easy way to create legend
    if save:
        plt.savefig(save + '/Forecast_Plot_ARIMA_' + name.title() + '_' + choice.title() + '.png')
    if show:
        plt.show()
    return predictions
    

def df_tidy(df, region):
    '''
    This function converts the input df into a df with the first three columns as 
    'country/state', 'lat', 'lon',
    and the rest of the columns as datestrings
    '''
    ncol = len(df.columns)
    nrow = len(df.index)
    index = date_index(df.columns)
    geo_dict = retrieve_geo_info(df, region)
    name, lat, lon = geo_dict["name"], geo_dict["lat"], geo_dict["lon"]
    df_new = df[[name, lat, lon]+list(df.columns[index[0]:(index[-1]+1)])]
    df_dict = {}
    for i in range(nrow):
        df_dict[i] = df_new[df_new.columns[0]][i]
    df_new.rename(index=df_dict, inplace = True)
    if region == "global":
        new_name = "country"
    elif region == "us":
        new_name = "state"
    else:
        new_name = region
    df_new.rename(columns = {name: new_name}, inplace = True)
    df_new.rename(columns = {lat: "lat"}, inplace = True)
    df_new.rename(columns = {lon: "lon"}, inplace = True)
    return df_new

#remove NaN rows
def remove_rows(df):
    rownames = df.index
    removed_indices = []
    column_names = df.columns
    lat = [s for s in column_names if "lat" == s.lower()[:3]]
    lon = [s for s in column_names if "lon" == s.lower()[:3]]

    if not lat:
        print(">>> ERROR: There must be a column indicateing the latitude of each region.")
        print("Exit...")
        exit(0)
    if not lon:
        print(">>> ERROR: There must be a column indicateing the longtitude of each region.")
        print(">>> Exit...")
        exit(0)
    lat, lon = lat[0], lon[0]
    df_lat = df[lat].to_numpy()
    df_lon = df[lon].to_numpy()
    for i in range(len(df)):
        name = df.index
        if np.isnan(df_lat[i]) or np.isnan(df_lon[i]):
            removed_indices.append(rownames[i])
    return df.drop(removed_indices, axis = 0)

# remove geographical info, leaving pure confirmed cases for each date
def remove_geo(df):
    df_new = df[match_date(df.columns)]
    return df_new

# check if a list of strings follows the default datestring format and return those that do match
def match_date(lst):
    mat = [s for s in lst if re.match('^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$', s) is not None]
    return mat

# check if a list of strings follows the default datestring format and return the index of those that do match
def date_index(lst):
    index = [i for i in range(len(lst)) if re.match('^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$', lst[i]) is not None]
    return index

# retrieve the 'country/state' column name, 'lat' column name, and 'lon' column name form a df
# region is 'global' or 'us'
def retrieve_geo_info(df, region):
    column_names = df.columns
    i, name = 0, ""
    while i < len(column_names):
        if region == "global":
            if "country" in column_names[i].lower():
                name = column_names[i]
                break
        if region == "us":
            if "state" in column_names[i].lower():
                name = column_names[i]
                break
        if region == "china":
            if "province" in column_names[i].lower():
                name = column_names[i]
                break
        i += 1
    lat = [s for s in column_names if "lat" == s.lower()[:3]]
    lon = [s for s in column_names if "lon" == s.lower()[:3]]
    lat, lon = lat[0], lon[0]
    if (not name) or (not lat) or (not lon):
        print(">>> ERROR: Country/state name, latitude, and lontitude columns needed in ", df)
        print(">>> Exit...")
        exit(0)
    return {"name": name, "lat": lat, "lon": lon}
    

# reference: https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/
def prediction_interval(model, params, X_train, y, number_new, alpha = 0.05, freq = None):
    ''' Compute a prediction interval around the model's prediction of 'number_new' new dates.
    INPUT
        model: function that takes X and params as parmaeters
        params: paramaters returned by curve_fit()
        X_train: Series Index (row names) from a dataframe
        y: Series Data (one row) from a dataframe
        number_new: number of new data points
        alpha: float = 0.05 
            The prediction uncertainty
        freq: freq param of pd.to_datetime() method

    OUTPUT
        A list of triples (`lower`, `pred`, `upper`) in the form of a number_new * 3 dataframe
        with `pred` being the prediction  of the model 
        and `lower` and `upper` constituting the lower- and upper 
        bounds for the prediction interval around `pred`, respectively. 
    '''

    # Number of training samples
    n = len(X_train)
    X = np.arange(n)
    x_new = np.array(range(n, n+number_new))
    # The authors choose the number of bootstrap samples as the square root 
    # of the number of samples
    nbootstraps = np.sqrt(n).astype(int)

    # Compute the m_i's and the validation residuals
    bootstrap_preds, val_residuals = np.zeros((nbootstraps,number_new)), []
    for b in range(nbootstraps):
        bootstrap_b = np.zeros((number_new))
        x_trains = np.random.choice(range(n), size = n, replace = True)
        x_vals = np.array([idx for idx in range(n) if idx not in x_trains])
        preds = model(x_vals, params[0], params[1], params[2])
        val_residuals.append(y[x_vals] - preds)
        bootstrap_preds[b, :] = model(x_new, params[0], params[1], params[2])
    bootstrap_preds -= np.mean(bootstrap_preds, axis = 0, keepdims = True)
    val_residuals = np.concatenate(val_residuals)

    # Compute the prediction and the training residuals
    preds = model(X, params[0], params[1], params[2])
    train_residuals = y - preds

    # Take percentiles of the training- and validation residuals to enable 
    # comparisons between them
    val_residuals = np.percentile(val_residuals, q = np.arange(100))
    train_residuals = np.percentile(train_residuals, q = np.arange(100))

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    no_information_error = np.mean(np.abs(np.random.permutation(y) - np.random.permutation(preds)))
    generalisation = np.abs(val_residuals - train_residuals)
    no_information_val = np.abs(no_information_error - train_residuals)
    relative_overfitting_rate = np.mean(generalisation / no_information_val)
    weight = .632 / (1 - .368 * relative_overfitting_rate)
    residuals = (1 - weight) * train_residuals + weight * val_residuals

    # Construct the C set and get the percentiles
    pred_intervals = np.zeros((number_new, 3))
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    for i in range(number_new):
        C = np.array([m + o for m in bootstrap_preds[i, ] for o in residuals])
        percentiles = np.percentile(C, q = qs)
        pred = model(x_new[i], params[0], params[1], params[2])
        pred_intervals[i, :] = [pred + percentiles[0], pred, pred + percentiles[1]]
    
    # convert numpy array to panda df
    New_Dates = pd.date_range(start = X_train[0], periods=number_new+n,freq=freq)
    pred_df = pd.DataFrame({'lower': pred_intervals[:,0],
                           'pred': pred_intervals[:,1],
                           'upper': pred_intervals[:,2]},
                          index = New_Dates[n:])
    return pred_df
    
# reference: https://medium.com/analytics-vidhya/how-to-predict-when-the-covid-19-pandemic-will-stop-in-your-country-with-python-d6fbb2425a9f
def logistic(X, c, k, m):
    return c/(1+np.exp(-k*(X-m)))

def gaussian(X, mu, sigma, c):
    return c*np.exp(-(1/2)*((X-mu)/sigma)**2)

def gaussian_stochastic(X, mu, sigma, c):
    return c*np.exp(-(1/2)*((X-mu)/sigma)**2) + np.random.normal(loc = 0.0, scale = np.sqrt(sigma), size = len(X))

def curve_fit(dataset, name, choice):
    '''
    fits the confirmed cases of a specific region given by 'name' (which are index names) in the 'dataset'
    the 'choice' has to be either total confirmed cases ('total') up to rach date or daily new cases ('new')
    '''
    X = np.arange(len(dataset.loc[name]))
    y = dataset.loc[name].values
    if choice == "total":
        logistic_model, cov = optimize.curve_fit(logistic,
                                xdata=X, 
                                ydata=y, 
                                maxfev=10000,
                                p0=[np.max(y), 1, 1])
        return logistic_model
    else:
        mle_mu = np.mean(y)
        mle_sigma = ((np.sum((y-np.mean(y))**2))/len(y))**0.5
        gaussian_model, cov = optimize.curve_fit(gaussian,
                                xdata=X, 
                                ydata=y, 
                                maxfev=10000,
                                p0=[mle_mu, mle_sigma, np.sum(y)])
        return gaussian_model
    

def forecast(dataset, region, name, choice, zoom = 30, figsize = (20, 5), pred_ahead = 10, freq = None, show = False, save = "."):
    '''
    this function both outputs a plot and returns a dataframe with 3 columns:
    'date', 'confirmed cases' and 'predicted cases';
    region: 'global'/'US'
    model_params are those that are returned by curve_fit(),
    pred_ahead is the number of days to be forecasted (must be > 0)
    freq is the frequency by which data wish to be presented.
    zoom: how many last datas what to be presented in the plot;
        e.g. zoom = 30 : plot the last 30 of the original data plus the new predicted data
    the other parameters are the same as in prediction_interval
    show: show the figure or not
    save: a directory in which the plot will be saved
    '''
    y = dataset.loc[name]
    current_dates = y.index
    New_Dates = pd.date_range(start=current_dates[0],periods=pred_ahead+len(current_dates),freq=freq)
    model_params = curve_fit(dataset, name, choice)
    X = np.arange(len(New_Dates))
    if choice == "total":
        fitted_y = logistic(X, model_params[0], model_params[1], model_params[2])
        prediction = prediction_interval(logistic, model_params,current_dates, y, number_new = pred_ahead, freq=freq)
    else:
        fitted_y = gaussian(X, model_params[0], model_params[1], model_params[2])
        prediction = prediction_interval(gaussian, model_params,current_dates, y, number_new = pred_ahead, freq=freq)
    
    df_new = pd.DataFrame(['NaN']*(pred_ahead), index = New_Dates[len(current_dates):])
    df_confirmed = pd.concat([y,df_new])
    df = pd.DataFrame({'Dates': New_Dates, 'Confirmed Cases': df_confirmed[0], 'Predicted Cases': fitted_y})
    dict_df = {}
    for i in range(len(New_Dates)):
        dict_df[df.index[i]] = i
    df.rename(index = dict_df, inplace = True)
    
    # plotting with prediction_intervals

    if show != "True":
        matplotlib.use("Agg") # stop popping up windows
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize=figsize)
    line1 = ax[0].plot(New_Dates, fitted_y, label = 'Predicted Cases', color = 'red')
    line2 = ax[1].plot(New_Dates[(len(current_dates)-zoom):], fitted_y[(len(current_dates)-zoom):], color = 'red', label = 'Predicted Cases')

    line3 = prediction['pred'].plot(ax=ax[0], grid=False, color="red")
    ax[0].fill_between(x=prediction.index, y1=prediction['lower'], y2=prediction['upper'], color='b', alpha=0.3)
    ax[0].set_ylim(0, max([max(y), max(fitted_y), max(prediction['upper'])])*1.05)
    line4 = prediction["pred"].plot(ax=ax[1], grid=False, color="red")
    ax[1].fill_between(x=prediction.index, y1=prediction['lower'], y2=prediction['upper'], color='b', alpha=0.3)
    ax[1].set_ylim(0, max([max(y[(len(current_dates)-zoom):]), max(fitted_y[(len(current_dates)-zoom):]), max(prediction['upper'])])*1.05)
    
    if choice == "total":
        line5 = ax[0].plot(pd.to_datetime(current_dates), y, 'ro', markersize = 2, label = 'Confirmed Cases', color = 'blue')
        ax[0].legend(line1 + line5, ['Predicted Cases', 'Confirmed Cases'], loc = 'best')
        line6 = ax[1].plot(pd.to_datetime(current_dates)[(len(current_dates)-zoom):], y[(len(current_dates)-zoom):], markersize = 2, label = 'Confirmed Cases')
        ax[1].legend(line2 + line6, ['Predicted Cases', 'Confirmed Cases'], loc = 'best')
    else:
        # create bar plots
        ax2 = ax[0].twinx()
        ax2.set_ylim(ax[0].get_ylim()) # so that the bar plot and the line graph habe the dame scale
        ax3 = ax[1].twinx()
        ax3.set_ylim(ax[1].get_ylim()) # so that the bar plot and the line graph habe the dame scale
        ax2.bar(pd.to_datetime(current_dates), y, align='center', alpha = 0.5, width = 0.5, label = 'Confirmed Cases', color = 'blue')
        ax2.grid(b=False)  
        ax2.axis("off") # turn off bar plot axis
        lines, labels = ax[0].get_legend_handles_labels()
        line, label = [lines[0]], [labels[0]] # make sure they are lists, do not retrieve the prediction interval line and label
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(line + lines2, label + labels2, loc='best', prop={'size': 10}) 
        ax3.bar(pd.to_datetime(current_dates)[(len(current_dates)-zoom):], y[(len(current_dates)-zoom):], align='center', alpha = 0.5, width = 0.5, label = 'Confirmed Cases', color = 'blue')
        ax3.grid(b=False)  
        ax3.axis("off") # turn off bar plot axis
        lines1, labels1 = ax[1].get_legend_handles_labels()
        line_1, label_1 = [lines1[0]], [labels1[0]]
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(line_1 + lines3, label_1 + labels3, loc='best', prop={'size': 10}) 
    
    fig.suptitle('Forecast Plot of COVID-19 Cases in' + region.title(), fontsize = 20)
    fig.text(0.27, 0.05, 'Date', fontsize = 15)
    fig.text(0.73, 0.05, 'Date', fontsize = 15)
    fig.text(0.07, 0.35, 'Number of ' + choice.title() + ' Cases', size = 15, rotation = 90)

    if save != "False":
        plt.savefig(save + '/Forecast_Plot_' + name.title() + '_' + choice.title() + '.png')
    if show == "True":
        plt.show()
    plt.close()    
    return df

# https://gist.github.com/rogerallen/1583593
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

china_province_map = {
    'Anhui': '安徽省',
    'Beijing': '北京市',
    'Chongqing': '重庆市',
    'Fujian': '福建省',
    'Gansu': '甘肃省',
    'Guangdong': '广东省',
    'Guangxi': '广西壮族自治区"',
    'Guizhou': '贵州省',
    'Hainan': '海南省',
    'Hebei': '河北省',
    'Heilongjiang': '黑龙江省',
    'Henan': '河南省',
    'Hong Kong': '香港特别行政区',
    'Hubei': '湖北省',
    'Hunan': '湖南省',
    'Inner Mongolia': '内蒙古自治区',
    'Jiangsu': '江苏省',
    'Jiangxi': '江西省',
    'Jilin': '吉林省',
    'Liaoning': '辽宁省',
    'Macau': '澳门特别行政区',
    'Ningxia': '宁夏回族自治区',
    'Qinghai': '青海省',
    'Shaanxi': '陕西省',
    'Shandong': '山东省',
    'Shanxi': '山西省',
    'Shanghai': '上海市',
    'Sichuan': '四川省',
    'Tianjin': '天津市',
    'Tibet': '西藏自治区',
    'Xinjiang': '新疆维吾尔自治区',
    'Yunnan': '云南省',
    'Zhejiang': '浙江省',
    'Taiwan': '台湾省'
}


# Map GIF
# https://towardsdatascience.com/covid-19-map-animation-with-python-in-5-minutes-2d6246c32e54
# https://towardsdatascience.com/how-to-create-an-animated-choropleth-map-with-less-than-15-lines-of-code-2ff04921c60b

def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

def plot_gif(dataset, region, choice, scale, save, show):
    '''
    this function displays an animated plot of the progress of COVID-19 cases over time
    dataset: a panda dataframe with the first 3 column as 'country'/'state', 'lat', and 'lon', 
            and the rest of the columns as dates
    region: 'global' or 'us'
    choice: 'total' or 'new'
    scale: 'continuous' or 'log'
    save: True or False
    '''
    # tidy data for analysis
    if match_date(dataset.columns) != dataset.columns[3:].tolist():
        print(dataset)
        print(">>> Make sure every column except the first 3 columns of your dataframe is a datestring in the format mm/dd/yy or mm/dd/yyyy")
        print(">>> Exit...")
        exit(0)
    
    if region == 'global':
        dataset_1st_column_name = 'country'
        dataset['iso_alpha_3'] = [get_country_code(name) for name in dataset['country']]
        df_long = pd.melt(dataset[[dataset.columns[0]]+list(dataset.columns[3:])], id_vars=['country','iso_alpha_3'])
    elif region == 'us':
        dataset_1st_column_name = 'state'
        dataset['abbr'] = [us_state_abbrev[state] for state in dataset['state']]
        df_long = pd.melt(dataset[[dataset.columns[0]]+list(dataset.columns[3:])], id_vars=['state','abbr'])
    elif region == 'china':
        dataset_1st_column_name = 'province'
        dataset['province'] = [china_province_map[name] for name in dataset['province']]
        df_long = pd.melt(dataset[[dataset.columns[0]]+list(dataset.columns[3:])], id_vars = 'province')
        df_long.rename(columns = {'province': 'name'}, inplace = True) 

    # check dataframe structure
    try: 
        dataset[dataset_1st_column_name]
        dataset['lat']
        dataset['lon']
    except KeyError as e:
        print('>>> The column names in your dataframe are: ', dataset.columns.tolist())
        print(">>> Make sure the first 3 columns of your dataframe are 'state', ", dataset_1st_column_name, ", 'lat', and 'lon'.")
        print(">>> Exit...")
        exit(0)

    # set color scales
    colors = df_long["value"]
    max_color_range = int(max(np.log10(colors)))
    if scale == 'log':
        range_color = (0, max_color_range)
        colors = np.log10(colors)
    else:
        range_color = (0, 10**max_color_range)

    # plot
    if region == 'global':
        fig = px.choropleth(df_long,                           # Input Dataframe
                            locations="iso_alpha_3",           # identify country code column
                            color=colors,                      # identify representing column
                            hover_name="country",              # identify hover name
                            animation_frame="variable",        # identify date column
                            projection="natural earth",        # select projection
                            color_continuous_scale = 'Peach',  # select prefer color scale
                            range_color=range_color,           # select range of dataset
                            title = 'COVID-19 ' + choice.title() + ' Confirmed Cases (Global)'
                            )
    elif region == 'us':
        fig = px.choropleth(df_long, 
                  locations = 'abbr',
                  color=colors, 
                  animation_frame="variable",
                  color_continuous_scale="Peach",
                  locationmode='USA-states',
                  scope="usa",
                  range_color=range_color,
                  title='COVID-19 Total Confirmed Cases (US)'
                    )
    elif region == 'china':
        # json file source: https://github.com/datasets/covid-19 converted to geojson using mapbuilder website
        data = json.load(open('china-geojson-master/china (geojson).json'))
        fig = px.choropleth_mapbox(df_long, 
                    geojson=data, 
                    locations='name', 
                    featureidkey="properties.name",
                    color=colors, 
                    animation_frame="variable",
                    color_continuous_scale="Peach",
                    range_color=range_color,
                    title='COVID-19 '+ choice.title() + ' Confirmed Cases (China)'
                )
        fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=2.5, mapbox_center = {"lat": 35.8617, "lon": 104.1954})
    
    # update color scale
    if scale == 'log':
        fig.update_layout(coloraxis_colorbar=dict(
                    tickvals=list(range(max_color_range)),
                    ticktext=[str(10**num) for num in range(max_color_range)]))
    
    # show and/or save
    if show == "True":
        print('Show map')
        fig.show()
    if save != False:          
        fig.write_html(save + '/' + region + '_map_' + choice + '_' + scale + 'scale.html')
    


def plot_alt(dataset, region, choice, scale, save, show, jsonfile):
    '''
    needs an additional geojson file to produce map
    '''
    try: 
        dataset['name']
    except KeyError as e:
        print('>>> The column names in your dataframe are: ', dataset.columns.tolist())
        print(">>> Make sure the first column of your dataframe is 'region'")
        print(">>> Exit...")
        exit(0)
    try:
        geojson = json.load(open(jsonfile))
    except FileNotFoundError as e:
        print(e)
        exit(0)
    try:
        names = [geojson['features'][i]['properties']['name'] for i in range(len(geojson))]
    except:
        try:
            test = geojson['features'][0]['properties']
        except:
            sample_data = json.load(open('china-geojson-master/china (geojson).json'))
            with open('sample_geojson(source website can be found in utils.py).json', 'w+') as f:
                json.dump(sample_data, f)
            print(">>> Make sure your geojson file follows the same format as 'sample_geojson.json'.")
            exit(0)
        true_key = False
        key_containing_name = []
        city_names = list(dataset['name'])
        for i in range(len(city_names)):
            for key in list(test.keys()):
                if test[key] == city_names[i]:
                    true_key = key
                    break
                if "name" in key.lower():
                    key_containing_name.append(key)
        if true_key != False:
            dataset.rename(columns = {'name': true_key}, inplace = True)
        else:
            print(">>> Make sure there is an entry in your json file that indicates the name of each region in the dataset.\nFor your reference, the city names are: " + str(city_names))
            exit(0)
    region_matches = []
    for name in dataset['regions'].tolist():
        if name not in names:
            print(">>> Your names of your geojson file do not match the names of your dataset.")
            exit(0)
            print(">>> Attempt to find closest match...")
            
    df_long = pd.melt(dataset[[dataset.columns[0]]+match_date(dataset.columns)], id_vars = name)
    colors = df_long["value"]
    max_color_range = int(max(np.log10(colors)))
    if scale == 'log':
        range_color = (0, max_color_range)
        colors = np.log10(colors)
    else:
        range_color = (0, 10**max_color_range)

    fig = px.choropleth_mapbox(df_long, 
                geojson=geojson, 
                locations=true_key, 
                featureidkey="properties."+true_key,
                color=np.log10(df_long["value"]), 
                animation_frame="variable",
                color_continuous_scale="Peach",
                range_color=(0, max_color_range),
                title='COVID-19 '+ choice.title() + ' Confirmed Cases (' + region + ')'
                )
    fig.update_layout(mapbox_style="carto-positron",
                  mapbox_zoom=2)
    if scale == 'log':
        fig.update_layout(coloraxis_colorbar=dict(
                        tickvals=list(range(max_color_range)),
                        ticktext=[str(10**num) for num in range(max_color_range)],))
    
    if show == "True":
        print('Show map') 
        fig.show()
    if save != False:
        fig.write_html(save + '/' + 'region_map_' + choice + '_' + scale + 'scale.html')   
    
    
    
    

