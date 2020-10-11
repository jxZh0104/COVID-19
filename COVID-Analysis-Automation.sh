#!/usr/bin/env bash
echo "This is an automation of COVID Analysis for creating maps."
echo "If you wish to interrupt execution at any time, press ctrl + Z"

read -p "Download latest data?: " response0
if [ $response0 = true ]
then
    echo ">>> Downloading data..."
    ./download-latest-JHU.sh
    Rscript COVID-proj.R
    echo ">>> Data download and preliminary cleaning complete."
fi

rm -rf automation
mkdir automation
mkdir automation/maps
echo ">>> Creating map gifs for world, us, and china data..."
for name in 'global_counts_total_geo' 'us_counts_total_geo' 'china_counts_total_geo' 'global_counts_new_geo' 'us_counts_new_geo' 'china_counts_new_geo' 
do  
    echo ">>> Plotting map gif for $name..." 
    for scale in 'log' 'continuous'
    do 
        python3 Analysis_Script.py --df $name --forecast False --map True --scale $scale --save automation/maps
    done
done
echo ">>> Map gifs saved!"

echo ">>> Creating forecast plots for all regions of world, us, and china data..."
mkdir -p automation/forecasts
for file in data_entries/*
do
    read -p ">>> Plot forecast plots for ${file:13:${#file}}?: " response2
    if [ $response2 = true ]
    then 
        let length=$((${#file}-17))
        mkdir -p automation/forecasts/${file:13:length}
        while IFS= read -r line
        do 
            # ${#string} gives the length of a string 
            python3 Analysis_Script.py --df ${file:13:$length} --forecast True --name $line --show False --save automation/forecasts/${file:13:length}
        done < $file
    fi
done
echo ">>> Forecast plots saved!"
echo ">>> Program complete. Exit..."