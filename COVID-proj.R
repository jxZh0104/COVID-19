## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

options(warn=-1)
## ----import-------------------------------------------------------------------
z1 = try(read.csv("jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", header = T, check.names = FALSE),
        silent = T)
if(inherits(z1, "try-error")) {
  stop("An error has occured when reading the data files.
       Please make sure that you have run the shell script to download data from jhu repo.")
}
z2 = try(read.csv("jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv", header = T, check.names = FALSE),
        silent = T)
if(inherits(z2, "try-error")) {
  stop("An error has occured when reading the data files.
       Please make sure that you have run the shell script to download data from jhu repo.")
}
covid_data_global =read.csv("jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", header = T, check.names = FALSE)
covid_data_us = read.csv("jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv", header = T, check.names = FALSE)

# reference: https://stackoverflow.com/questions/18931006/how-to-suppress-warning-messages-when-loading-a-library
suppressWarnings(suppressMessages(library(tidyverse)))

o = order(as.character(covid_data_global$`Country/Region`))
covid_data_global = covid_data_global[o, ]
o2 = order(as.character(covid_data_us$Province_State))
covid_data_us = covid_data_us[o2, ]


## ----gather-------------------------------------------------------------------
country_names = table(covid_data_global$`Country/Region`)
country_times = country_names
names(country_times) = NULL
m = length(country_names)
n = ncol(covid_data_global)
lst_country = vector(mode = 'list', length = as.integer(m))
names(lst_country) = names(country_names)
index = 0
for (i in 1:m){
  lst_country[[i]] = (index+1):(index + country_times[i])
  index = index + country_times[i]
}
total_counts_global = sapply(1:m, function(i){
  colSums(covid_data_global[lst_country[[i]], 5:n])
}) %>% t() %>% as.data.frame()
colnames(total_counts_global) = colnames(covid_data_global)[5:n]
rownames(total_counts_global) = names(country_names)

state_names = table(covid_data_us$Province_State)
state_times = state_names
m2 = length(state_names)
n2 = ncol(covid_data_us)
lst_state = vector(mode = 'list', length = as.integer(m2))
names(lst_state) = names(state_names)
index2 = 0
for (i in 1:m2){
  lst_state[[i]] = (index2+1):(index2 + state_times[i])
  index2 = index2 + state_times[i]
}
total_counts_us = sapply(1:m2, function(i){
  colSums(covid_data_us[lst_state[[i]], 12:n2])
}) %>% t() %>% as.data.frame()
colnames(total_counts_us) = colnames(covid_data_us)[12:n2]
rownames(total_counts_us) = names(state_names)
print('Total confirmed cases gathered!')


## ----map----------------------------------------------------------------------
# data source: https://www.kaggle.com/eidanch/counties-geographic-coordinates
# data source: https://www.kaggle.com/washimahmed/usa-latlong-for-state-abbreviations
us_map_coord = read.csv("us_states.csv", header = T)
world_map_coord = read.csv("countries.csv", header = T)
add_geographical_data <- function(df_input, region){
  df = df_input
  df$lat = vector(mode = "numeric", length = as.integer(nrow(df)))
  df$lon = vector(mode = "numeric", length = as.integer(nrow(df)))
  if (region == 'world'){
    for (i in 1:nrow(df)){
      index1 = which(rownames(df)[i] == world_map_coord$country)
      index2 = which(rownames(df)[i] == world_map_coord$name)
      if (length(index1) == 1){
        df$lat[i] = world_map_coord$latitude[index1]
        df$lon[i] = world_map_coord$longitude[index1]
      } else {
        if (length(index2) == 1){
          df$lat[i] = world_map_coord$latitude[index2]
          df$lon[i] = world_map_coord$longitude[index2]
        } else {
          df$lat[i] = NA
          df$lon[i] = NA
        }
      }
    }
  }else{
    for (i in 1:nrow(df)){
      index = which(rownames(df)[i] == us_map_coord$City)
      if (length(index) == 1){
        df$lat[i] = us_map_coord$Latitude[index]
        df$lon[i] = us_map_coord$Longitude[index]
      } else {
        df$lat[i] = NA
        df$lon[i] = NA
      }
    }
  }
  return(df)
}

country_names_actual = names(country_names)
state_names_actual = names(state_names)

total_counts_global$country = country_names_actual
total_counts_us$state = state_names_actual
total_counts_global = add_geographical_data(total_counts_global, 'world')
total_counts_us = add_geographical_data(total_counts_us, 'us')

print('Longtitudes and latitudes added!')


## ----new----------------------------------------------------------------------
new_counts_global = total_counts_global
for (i in 2:(n-4)){
  new_counts_global[,i] = total_counts_global[,i] - total_counts_global[, i-1]
}

new_counts_us = total_counts_us
for (i in 2:(n2-11)){
  new_counts_us[,i] = total_counts_us[,i] - total_counts_us[, i-1]
}

print('New counts dfs created!')
## ----save---------------------------------------------------------------------
save(total_counts_global, total_counts_us, new_counts_global, new_counts_us, 
     file = "COVID-19_cleaned_cases.RData")
print('RData file saved!')

