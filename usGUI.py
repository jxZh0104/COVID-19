from ClassGUI import GUI
from tkinter import *
import sqlite3

US_GUI = GUI(640, "US", Tk(), sqlite3.connect("sqlite3/COVID_database.db"))
US_GUI.run()
