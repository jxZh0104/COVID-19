# COVID-19 Analysis Pipeline

0. Navigate to the current folder and run 'chmod +x download-latest-JHU.sh' to make the shell script file executable. <br/>
1. Run './download-latest-JHU.sh' in terminal to gather latest data. <br/>
2. Run 'Rscript COVID-proj.R' in terminal for preliminary cleaning and gathering of gathered data. <br/>
3. Run 'python3 COVID-proj.py' in terminal to analyze and plot data (make sure all the necessary modules have already been installed). <br/>
If you don't want RuntimeWarning to be printed, change step 3 to:<br/>
3. Run 'python3 -W ignore COVID-proj.py' in terminal to analyze and plot data.<br/>
If you wish to personalize which data you want to analyze, run 'python3 Analysis_Script.py -h' to see the necessary arguments to input. <br/>

OR

Simply run 
'chmod +x COVID-Analysis.sh'
followed by 
'./COVID-Analysis.sh'.

'GUI.py' generates a simple GUI for querying number of confirmed cases and can plot cases in Berkeley.<br/>
‘usGUI.py' generates a simple GUI for querying number of confirmed cases in different states of US and can plot them.<br/>
'globalGUI.py' generates a simple GUI for querying number of confirmed cases in different countries and can plot them.
