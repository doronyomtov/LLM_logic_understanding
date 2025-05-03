import customtkinter as ctk
import os
import webbrowser
from experiment_page import open_experiment_page


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

def open_pdf():
    """Open the research PDF file."""
    pdf_path = os.path.abspath("research.pdf")
    if os.path.exists(pdf_path):
        webbrowser.open(f"file://{pdf_path}")
    else:
        ctk.CTkMessagebox(title="Error", message="research.pdf not found.", icon="cancel")

app = ctk.CTk()
app.title("LLM Logic")
app.geometry("800x500")
app.resizable(False, False)

title_icon = ctk.CTkLabel(app, text="🌐", font=ctk.CTkFont(size=48))
title_icon.pack(pady=(40, 10))

main_title = ctk.CTkLabel(app, text="LLM LOGIC", font=ctk.CTkFont(size=36, weight="bold"), text_color="#6a5acd")
main_title.pack()

subtitle = ctk.CTkLabel(
    app,
    text="Exploring the reasoning capabilities and performance of large language\n"
         "models through structured experiments and comprehensive research",
    font=ctk.CTkFont(size=14),
    justify="center",
    text_color="#444444"
)
subtitle.pack(pady=(10, 30))


button_frame = ctk.CTkFrame(app, fg_color="transparent")
button_frame.pack(pady=10)

btn_experiment = ctk.CTkButton(
    master=button_frame,
    text="Start Experiment",
    command=lambda: open_experiment_page(app),
    font=ctk.CTkFont(size=14, weight="bold"),
    fg_color="#4a68ff",
    hover_color="#3a50cc",
    text_color="white",
    width=160,
    height=40,
    corner_radius=10
)
btn_experiment.grid(row=0, column=0, padx=20)

btn_research = ctk.CTkButton(
    master=button_frame,
    text="View Research",
    command=open_pdf,
    font=ctk.CTkFont(size=14, weight="bold"),
    fg_color="white",
    text_color="black",
    hover_color="#e0e0e0",
    width=160,
    height=40,
    corner_radius=10,
    border_width=1,
    border_color="#999"
)
btn_research.grid(row=0, column=1, padx=20)

app.mainloop()
