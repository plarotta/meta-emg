import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import json


class DataCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Collector")
        self.root_dir = '/Users/plarotta/software/meta-emg/data/collected_data'

        self.task_collection = []

        self.session_index = 0
        self.current_session = None
        self.current_csv_files = None

        self.all_sessions = [session for session in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, session))]
        self.all_index = 0
    

        # Create GUI components
        self.session_label = tk.Label(root, text="Session:")
        self.session_label.pack()

        # Create a Figure and Axes objects
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.info_label = tk.Label(root, text="")
        self.info_label.pack(pady=10)

        self.approve_button = tk.Button(root, text="Approve", command=self.approve_files)
        self.approve_button.pack(side=tk.LEFT, padx=10)

        self.remove_button = tk.Button(root, text="Undo", command=self.remove_last_task)
        self.remove_button.pack(side=tk.LEFT, padx=10)

        self.back_button = tk.Button(root, text="Back", command=self.load_previous_csv_file)
        self.back_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(root, text="Next", command=self.load_next_csv_file)
        self.next_button.pack(side=tk.LEFT, padx=10)

         # Create a Save & Exit button
        self.save_exit_button = tk.Button(root, text="Save & Exit", command=self.save_and_exit)
        self.save_exit_button.pack(side=tk.LEFT, padx=10)

        # Set up initial state
        self.load_next_session()

    def load_next_session(self):
        session_path = self.get_next_session()
        if session_path:
            self.current_session = session_path
            self.current_csv_files = self.get_csv_files(session_path)
            self.session_index = 0
            self.show_csv_file()

    def load_previous_csv_file(self):
        if self.current_csv_files:
            # Move to the previous CSV file within the current session
            if self.session_index -1 >= 0:
                self.session_index -= 1
                self.show_csv_file()

    def load_next_csv_file(self):
        if self.current_csv_files:
            # Move to the next CSV file within the current session
            self.session_index += 1
            if self.session_index < len(self.current_csv_files):
                self.show_csv_file()
            else:
                # If no more files in the current session, move to the next session
                self.load_next_session()

    def show_csv_file(self):
        # Clear the current plot
        self.ax.clear()

        csv_file = self.current_csv_files[self.session_index]
        df = pd.read_csv(csv_file)

        # Display the data (you can customize this part based on your needs)
        
        [self.ax.plot(df['time_elapsed'], df['emg'+str(idx)]) for idx in range(8)]
        self.ax.plot(df['time_elapsed'], df['gt']*.5*max(df['emg3']))
        self.ax.set_title("EMG Recording")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("EMG Signal")

        # Update the info label with the current session, CSV file names, and index information
        session_name = os.path.basename(self.current_session)
        file_name = os.path.basename(csv_file)
        self.info_label.config(text=f"Session: {session_name}\nFile: {file_name}\nIndex: {self.session_index + 1}/{len(self.current_csv_files)}")

        # Redraw the canvas
        self.canvas_frame.destroy()
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def approve_files(self):
        if self.current_session and self.current_csv_files:
            # Save the session name and approved CSV files to a file or data structure
            # You can customize this part based on how you want to store the approved data
            csv_filename = os.path.basename(self.current_csv_files[self.session_index])
            self.task_collection.append({'session': os.path.basename(self.current_session), 
                                         'condition': re.search(r'_(.*?)\.csv', csv_filename).group(1)})
            
            print("task collection updated... here's what it looks like now:", self.task_collection)

            # Load the next CSV file within the current session
            self.load_next_csv_file()

    def remove_last_task(self):
        if self.task_collection:
            removed_task = self.task_collection.pop()
            print(f"Removed Task: {removed_task}")

            self.load_next_csv_file()

    def get_next_session(self):
        # You can implement your logic to get the next session
        # For example, using os.listdir, os.path.join, and checking if it's a directory
        # Return None if there are no more sessions
        next_sess = os.path.join(self.root_dir, self.all_sessions[self.all_index])
        self.all_index += 1
        return next_sess

    def get_csv_files(self, session_path):
        # You can implement your logic to get the CSV files in the session directory
        # For example, using os.listdir and checking if it's a CSV file
        csv_files = [os.path.join(session_path, file) for file in os.listdir(session_path) if file.endswith('.csv')]
        return csv_files
    
    def save_and_exit(self):
        # Prompt the user for a file name to save the task collection
        if len(self.task_collection) < 1:
            plt.close(self.fig)
            self.root.destroy()
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])

            if file_path:
                # Save self.task_collection to the specified JSON file
                with open(file_path, 'w') as json_file:
                    json.dump(self.task_collection, json_file)
                
                plt.close(self.fig)

                # Close the GUI
                self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorGUI(root)
    root.mainloop()
