import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk

app = tk.Tk()
app.geometry("532x622")
app.title("Molecule Generator")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, fg_color="white", text_color="black")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10,y=110)

trigger = ctk.CTkButton(master=app, height=40, width=120, fg_color="white", text_color="blue")
trigger.configure(text="Generar Cadena")
trigger.place(x=206, y=60)

app.mainloop()
