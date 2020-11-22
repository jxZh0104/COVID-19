#!/usr/bin/env bash
echo ">>> Welcome! Please follow the directions given by the prompts below."

read -p ">>> Download data?: " data
if [ $data = true ]
then 
    ./download-latest-JHU.sh
    Rscript COVID-proj.R
    echo ">>> Data download and preliminary cleaning complete."
fi

read -p ">>> Update database?: " update
if [ $update = true ]
then 
    python3 database.py
    echo ">>> Database update complete."
fi

read -p ">>> Manual Check on Statistics With GUI?: " check
if [ $check = true ]
then 
    echo ">>> Preparing GUI..."
    python3 GUI.py
fi

read -p ">>> Daily Check on CA Cases?: " dailyCheck
if [ $dailyCheck = true ]
then 
    echo ">>> Generating US daily confirmed cases data..."
    python3 Analysis_Script.py --df us_counts_new --print True
    echo ">>> Generating forecast plot for California..."
    python3 Analysis_Script.py --df us_counts_new --name California --forecast True --show True --arima True
    echo ">>> Daily data checking and forecasting complete."
fi

read -p ">>> Daily Check on US Cases?: " dailyCheckUs
if [ $dailyCheckUs = true ]
then 
echo ">>> Generating global daily confirmed cases data..."
    python3 Analysis_Script.py --df global_counts_new --print True
    echo ">>> Generating forecast plot for US..."
    python3 Analysis_Script.py --df global_counts_new --name US --forecast True --show True --arima True
    echo ">>> Daily data checking and forecasting complete."
else
    # source: https://tecadmin.net/use-logical-or-and-in-shell-script/
    read -p ">>> Run python script?: " zeroth
    if [ $zeroth = true ]
    then
        read -p ">>> Enter dataset name: " first
        read -p ">>> Enter forecast (True or False): " second
        if [ $second = "True" ]
        then
            read -p ">>> Enter region name for analysis: " third
            read -p ">>> Use ARIMA for forecast? (True or False): " fourth
            read -p ">>> Show interactive plot? (True or False): " fifth
            python3 Analysis_Script.py --df $first --forecast $second --name $third --arima $fourth --show $fifth
        else
            read -p ">>> Enter map (True or False): " fourth
            if [ $third = "True" ]
            then 
                read -p ">>> Enter map color scale (log or continuous): " fourth
                read -p ">>> Enter directory for map to be saved in (input '.' if want to save in the current directory) or False: " fifth
                read -p ">>> Show plots (Enter True) or do not show (Enter False): " sixth
                python3 Analysis_Script.py --df $first --name $second --map $third --scale $fourth --save $fifth --show $sixth
            fi
        fi
    else
        echo ">>> Exit..."
    fi
fi
echo ">>> Program complete. Have a nice day!"