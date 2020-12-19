from tkinter import *
import sqlite3
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

root = Tk()
root.title("Simple Query GUI for COVID-19 Data")
root.geometry("640x640")

conn = sqlite3.connect("sqlite3/COVID_database.db")
c = conn.cursor()

tableNamePart1 = Entry(root, width = 40)
tableNamePart1.grid(row = 0, column = 1, padx = 20)
tableNamePart1_label = Label(root, text = "Region")
tableNamePart1_label.grid(row = 0, column = 0)

rowName = Entry(root, width = 40)
rowName.grid(row = 1, column = 1, padx = 20)
rowName_label = Label(root, text = "Subregion")
rowName_label.grid(row = 1, column = 0)

tableNamePart2 = Entry(root, width = 40)
tableNamePart2.grid(row = 2, column = 1, padx = 20)
tableNamePart2_label = Label(root, text = "New / Total")
tableNamePart2_label.grid(row = 2, column = 0)

date = Entry(root, width = 40)
date.grid(row = 3, column = 1, padx = 20)
date_label = Label(root, text = "Date (mm-dd-yy)")
date_label.grid(row = 3, column = 0)

query_label = Label(root, text = "")
query_label.grid(row = 5, column = 0, columnspan = 2)

def query():
    regionName = tableNamePart1.get().lower()
    tableName = regionName + "_counts_" + tableNamePart2.get().lower()
    dateEntry = "Date" + date.get().replace("-", "")
    subregion = rowName.get()

    global query_label
    query_label.destroy()

    if regionName == "china":
        firstColumn = "province"
    elif regionName == "global":
        firstColumn = "country"
    elif regionName == "us":
        firstColumn = "state"
    else:
        if regionName != "berkeley":
            query_label = Label(root, text = "Please enter region as one of 'China', 'Global', or 'US'")
            query_label.grid(row = 5, column = 0, columnspan = 2)
            return
    if regionName != "berkeley":
        command = "SELECT " + dateEntry + " FROM " + tableName + " WHERE " +  firstColumn + " = '" + subregion + "';"
    else:
        subregion = "Berkeley"
        if (date.get() == ""):
            query_label = Label(root, text = "Please enter a date")
        else:
            dateEntry = "20" + date.get()[6:] + "-" + date.get()[0:2] + "-" + date.get()[3:5]
            command = "SELECT " + tableNamePart2.get().title() + " FROM Berkeley WHERE Date = '" + dateEntry + "';"

    try:
        c.execute(command)
    except sqlite3.OperationalError as e:
        query_label = Label(root, text = e)
        query_label.grid(row = 5, column = 0, columnspan = 2)
        return

    records = c.fetchone()
    if records == None:
        query_label = Label(root, text = "No record matched your query")
    else:
        query_label = Label(root, text = tableNamePart2.get() + " confirmed cases in " + subregion + " on " + date.get() + ": " + str(records[0]))
    query_label.grid(row = 5, column = 0, columnspan = 2)

    '''
    tableNamePart1.delete(0, END)
    tableNamePart2.delete(0, END)
    rowName.delete(0, END)
    date.delete(0, END)
    '''
    return

HUD_label = Label(root, text = " ")
plot_label = Label(root, text = " ")
HUD_label.grid(row = 7, column = 0, columnspan = 2)

def on_click(event, ax, minDate, maxDate, minCount, maxCount):
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    timeDelta = maxDate - minDate
    def toDate(x):
        gradient = timeDelta.days/(xmax-xmin)
        normalized = int(gradient * (x-xmin))
        currDate = timedelta(days = normalized) + minDate
        return currDate
    def toData(y):
        gradient = (maxCount-minCount)/(ymax-ymin)
        normalized = minCount + gradient * (y-ymin)
        return normalized
    global HUD_label
    HUD_label.destroy()
    if event.inaxes is not None:
        # print("This function is not complete")
        HUD_label = Label(root, text = "x: " + str(toDate(event.xdata)) + " y: " + str(round(toData(event.ydata),1)))
        HUD_label.grid(row = 7, column = 0, columnspan = 2)
    else:
        HUD_label = Label(root, text = "                           ")
        HUD_label.grid(row = 7, column = 0, columnspan = 2)


def plotBerkeley():
    global plot_label
    if tableNamePart2.get() == '':
        plot_label = Label(root, text = "Please enter New/Total")
        plot_label.grid(row = 7, column = 0, columnspan = 2)
        return
    plot_label.destroy();
    command1 = "SELECT " + tableNamePart2.get().title() + " FROM Berkeley;"
    c.execute(command1)
    data = [r[0] for r in c.fetchall()]
    command2 = "SELECT Date FROM Berkeley;"
    c.execute(command2)
    date = [parse(d[0]).date() for d in c.fetchall()]
    fig = plt.figure(figsize=(6, 5), dpi=60)
    ax = fig.add_subplot(111)
    xmin = date[0] - timedelta(days = 5)
    xmax = date[-1] + timedelta(days = 5)
    ax.set_xlim([xmin, xmax])
    yrange = max(data) - min(data)
    ymin = min(data) - yrange / 10
    ymax = max(data) + yrange / 10
    ax.set_ylim([ymin, ymax])
    ax.plot(date, data)
    plt.title(tableNamePart2.get().title() + " Cases in Berkeley")

    # @source: https://stackoverflow.com/questions/59550783/embedding-a-matplotlib-graph-in-tkinter-grid-method-and-customizing-matplotl
    # specify the root as master
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=8, column=0, columnspan = 2, ipadx=100, ipady=20)

    # @source: https://matplotlib.org/3.1.1/users/event_handling.html
    fig.canvas.mpl_connect('motion_notify_event', lambda event: on_click(event, ax, xmin, xmax, ymin, ymax))

    # navigation toolbar
    '''
    toolbarFrame = Frame(master=root)
    toolbarFrame.grid(row=8,column=0)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    '''
    

query_button = Button(root, text = "Check COVID-19 Cases", command = query)
query_button.grid(row = 4, column = 0, columnspan = 2, pady = 20, padx = 20, ipadx = 200)

plot_button = Button(root, text = "Plot Berkeley Cases", command = plotBerkeley)
plot_button.grid(row = 6, column = 0, columnspan = 2, pady = 20, padx = 20, ipadx = 200)

root.mainloop()
conn.commit()
conn.close()
