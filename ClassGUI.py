from tkinter import *
import sqlite3
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class GUI:

    def __init__(self, dim, region, root, conn):
        assert isinstance(dim, int)
        assert isinstance(region, str)
        assert region in ["GLOBAL", "US", "CHINA"]
        self.dim = dim
        self.region = region
        self.root = root
        self.conn = conn
        self.rowName = Entry(root, width = 40)
        self.tableNamePart2 = Entry(root, width = 40)
        self.date = Entry(root, width = 40)
        self.query_label = Label(root)
        self.HUD_label = Label(root, text = "                             ")
        self.plot_label = Label(root, text = "                             ")
        self.c = conn.cursor()
        self.query_button = Button(self.root, text = "Check COVID-19 Cases", command = self.query)
        self.plot_button = Button(self.root, text = "Plot " + region + " Cases", command = self.plot)
        self.tableNameNew = self.region.lower() + "_counts_" + "new"
        self.tableNameTotal = self.region.lower() + "_counts_" + "total"
        self.c.execute("PRAGMA table_info(" + self.tableNameNew + ")")
        records1 = self.c.fetchall()
        self.datesNew = ["20" + r[1][8:10] + "/" + r[1][4:6] + "/" + r[1][6:8] for r in records1[3:]] # convert to yyyy/mm/dd form
        self.selectColumnCommandsNew = "SELECT " + ",".join([r[1] for r in records1[3:]]) + " FROM " + self.tableNameNew #+ " WHERE state = 'California';"
        self.c.execute("PRAGMA table_info(" + self.tableNameTotal + ")")
        records2 = self.c.fetchall()
        self.datesTotal = ["20" + r[1][8:10] + "/" + r[1][4:6] + "/" + r[1][6:8] for r in records2[3:]]
        self.selectColumnCommandsTotal = "SELECT " + ",".join([r[1] for r in records2[3:]]) + " FROM " + self.tableNameTotal #+ " WHERE state = 'California';"
        root.geometry(str(dim)+"x"+str(dim))

    def initializeGUI(self):
        self.root.title("Simple Query GUI for " + self.region + " COVID-19 Data")

        self.rowName.grid(row = 0, column = 1, padx = 20)
        self.rowName_label = Label(self.root, text = "Subregion")
        self.rowName_label.grid(row = 0, column = 0)

        self.tableNamePart2.grid(row = 1, column = 1, padx = 20)
        self.tableNamePart2_label = Label(self.root, text = "New / Total")
        self.tableNamePart2_label.grid(row = 1, column = 0)

        self.date.grid(row = 2, column = 1, padx = 20)
        self.date_label = Label(self.root, text = "Date (mm-dd-yy)")
        self.date_label.grid(row = 2, column = 0)

        self.query_label.grid(row = 4, column = 0, columnspan = 2)
        self.HUD_label.grid(row = 6, column = 0, columnspan = 2)
        self.plot_label.grid(row = 7, column = 0, columnspan = 2)

        self.query_button.grid(row = 3, column = 0, columnspan = 2, pady = 20, padx = 20, ipadx = 200)
        self.plot_button.grid(row = 5, column = 0, columnspan = 2, pady = 20, padx = 20, ipadx = 200)


    def processInfo(self, plot):
        if self.tableNamePart2.get().lower() == "new":
            tableName = self.tableNameNew
            tempCommand = self.selectColumnCommandsNew
        elif self.tableNamePart2.get().lower() == "total":
            tableName = self.tableNameTotal
            tempCommand = self.selectColumnCommandsTotal
        else:
            self.query_label = Label(self.root, text = "Please enter New/Total")
            self.query_label.grid(row = 4, column = 0, columnspan = 2)
            return {}
        tableName = self.region.lower() + "_counts_" + self.tableNamePart2.get().lower()
        subregion = self.rowName.get()
        dateEntry = "Date" + self.date.get().replace("-", "")

        if self.region == "CHINA":
            firstColumn = "province"
        elif self.region == "GLOBAL":
            firstColumn = "country"
        elif self.region == "US":
            firstColumn = "state"
        if plot == False:
            if (self.date.get() == ""):
                self.query_label = Label(self.root, text = "Please enter a date")
                self.query_label.grid(row = 4, column = 0, columnspan = 2)
                return {}
            else:
                dateEntry = "20" + self.date.get()[6:] + "-" + self.date.get()[0:2] + "-" + self.date.get()[3:5]
                command = "SELECT " + dateEntry + " FROM " + tableName + " WHERE " +  firstColumn + " = '" + subregion + "';"
        else:
            command = tempCommand + " WHERE " + firstColumn + " = '" + subregion + "';"

        return {"tableName": tableName, "command": command, "queryLabelPart1": self.tableNamePart2.get().title() + " confirmed cases in " + subregion + " on " + self.date.get() + ": ",
        "tableNamePart2Input": self.tableNamePart2.get()}

    def query(self):
        self.HUD_label.destroy()
        print("loading...")
        info = self.processInfo(False)
        print(info)
        if len(info) == 0:
            return
        command = info["command"]
        try:
            self.c.execute(command)
        except sqlite3.OperationalError as e:
            self.query_label = Label(self.root, text = e)
            self.query_label.grid(row = 4, column = 0, columnspan = 2)
            return
        records = self.c.fetchone()
        if records == None:
            self.query_label = Label(self.root, text = "No record matched your query")
        else:
            self.query_label = Label(self.root, text = info["queryLabelPart1"] + str(records[0]))
            #print(info["queryLabelPart1"] + str(records[0]))
        self.query_label.grid(row = 4, column = 0, columnspan = 2)

        '''
        tableNamePart1.delete(0, END)
        tableNamePart2.delete(0, END)
        rowName.delete(0, END)
        date.delete(0, END)
        '''
        return

    def on_click(self, event, ax, minDate, maxDate, minCount, maxCount):
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
        if event.inaxes is not None:
            # print("This function is not complete")
            self.HUD_label = Label(self.root, text = "x: " + str(toDate(event.xdata)) + " y: " + str(round(toData(event.ydata),1)))
            self.HUD_label.grid(row = 6, column = 0, columnspan = 2)
        else:
            self.HUD_label = Label(self.root, text = "                                           ")
            self.HUD_label.grid(row = 6, column = 0, columnspan = 2)


    def plot(self):
        if self.tableNamePart2.get().lower() not in ["new", "total"]:
            self.plot_label = Label(self.root, text = "Please enter New/Total")
            self.plot_label.grid(row = 7, column = 0, columnspan = 2)
            return
        self.plot_label.destroy()
        self.HUD_label.destroy()
        info = self.processInfo(True);
        if len(info) == 0:
            return
        command1 = info["command"]
        #print(command1)
        self.c.execute(command1)
        data = self.c.fetchall()[0]
        if self.tableNamePart2.get().lower() == "new":
            date = [parse(d).date() for d in self.datesNew]
        else:
            date = [parse(d).date() for d in self.datesTotal]
        fig = plt.figure(figsize=(6, 5), dpi=60)
        ax = fig.add_subplot(111)
        xmin = date[0] - timedelta(days = 5)
        xmax = date[-1] + timedelta(days = 5)
        ax.set_xlim([xmin, xmax])
        #print(data)
        #print(date)
        yrange = max(data) - min(data)
        ymin = min(data) - yrange / 10
        ymax = max(data) + yrange / 10
        ax.set_ylim([ymin, ymax])
        ax.plot(date, data)
        plt.title(self.tableNamePart2.get().title() + " Cases in " + self.rowName.get().upper())

        # @source: https://stackoverflow.com/questions/59550783/embedding-a-matplotlib-graph-in-tkinter-grid-method-and-customizing-matplotl
        # specify the root as master
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=0, columnspan = 2, ipadx=100, ipady=20)

        # @source: https://matplotlib.org/3.1.1/users/event_handling.html
        fig.canvas.mpl_connect('motion_notify_event', lambda event: self.on_click(event, ax, xmin, xmax, ymin, ymax))

        # navigation toolbar
        '''
        toolbarFrame = Frame(master=root)
        toolbarFrame.grid(row=8,column=0)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
        '''
        
    def run(self):
        self.initializeGUI()
        self.root.mainloop()
        self.conn.commit()
        self.conn.close()
