from tkinter import *
import sqlite3

root = Tk()
root.title("Simple Query GUI for COVID-19 Data")
root.geometry("600x400")

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
        query_label = Label(root, text = "Please enter region as one of 'China', 'Global', or 'US'")
        query_label.grid(row = 5, column = 0, columnspan = 2)
        return
    command = "SELECT " + dateEntry + " FROM " + tableName + " WHERE " +  firstColumn + " = '" + subregion + "';"

    try:
        c.execute(command)
    except sqlite3.OperationalError as e:
        query_label = Label(root, text = e)
        query_label.grid(row = 5, column = 0, columnspan = 2)
        return

    records = c.fetchone()
    query_label = Label(root, text = tableNamePart2.get() + " confirmed cases in " + subregion + " on " + date.get() + ": " + str(records[0]))
    query_label.grid(row = 5, column = 0, columnspan = 2)

    tableNamePart1.delete(0, END)
    tableNamePart2.delete(0, END)
    rowName.delete(0, END)
    date.delete(0, END)
    return

query_button = Button(root, text = "Check COVID-19 Cases", command = query)
query_button.grid(row = 4, column = 0, columnspan = 2, pady = 20, padx = 20, ipadx = 100)

root.mainloop()
conn.commit()
conn.close()
