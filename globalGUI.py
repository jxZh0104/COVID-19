from ClassGUI import GUI
from tkinter import *
import sqlite3

GLOBAL_GUI = GUI(640, "GLOBAL", Tk(), sqlite3.connect("sqlite3/COVID_database.db"))
GLOBAL_GUI.run()
