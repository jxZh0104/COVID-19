#!/usr/bin/env bash
mkdir -p automation/forecasts
for file in data_entries/*
do
    read -p "Retrieve $file?: " response
    echo $file
    if [ $response = true ]
    then 
        let length=$((${#file}-17))
        mkdir -p automation/forecasts/${file:13:length}
        while IFS= read -r line
        do 
            echo $line
            # ${#string} gives the length of a string 
            python3 Analysis_Script.py --df ${file:13:$length} --forecast True --name $line --show False --save automation/forecasts/${file:13:length}
        done < $file
    fi
done
