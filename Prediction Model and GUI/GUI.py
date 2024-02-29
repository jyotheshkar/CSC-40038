# Required Imports
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import math
import FinalModel as fm
import FutureWorkPrediction as fwd
# Define class called App
class App:
     # Initialize the App class
    def __init__(self, root):
        self.root = root
        # Create a title
        self.root.title("Event Booking Prediction")
       
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)
       
        self.create_widgets()
    # Define method called create_widgets for creating widgets
    def create_widgets(self):
        # Create label and entry widget for Start Date
        self.start_date_label = tk.Label(self.frame, text="Start Date (dd/mm/yyyy):")
        self.start_date_label.grid(row=0, column=0, sticky="e")
        self.start_date_entry = tk.Entry(self.frame)
        self.start_date_entry.grid(row=0, column=1, pady=(0, 5))
       
        # Create label and entry widget for End Date
        self.event_date_label = tk.Label(self.frame, text="Event Date (dd/mm/yyyy):")
        self.event_date_label.grid(row=1, column=0, sticky="e")
        self.event_date_entry = tk.Entry(self.frame)
        self.event_date_entry.grid(row=1, column=1, pady=(0, 5))
       
        # Create label and entry widget for Current Date
        self.current_date_label = tk.Label(self.frame, text="Current Date (dd/mm/yyyy):")
        self.current_date_label.grid(row=2, column=0, sticky="e")
        self.current_date_entry = tk.Entry(self.frame)
        self.current_date_entry.grid(row=2, column=1, pady=(0, 5))
       
        # Create label and entry widget for Number of bookings
        self.booking_label = tk.Label(self.frame, text="Number of Bookings:")
        self.booking_label.grid(row=3, column=0, sticky="e")
        self.validate_booking = (self.root.register(self.on_validate_booking), '%P')
        self.booking_entry = tk.Entry(self.frame, validate="key", validatecommand=self.validate_booking)
        self.booking_entry.grid(row=3, column=1, pady=(0, 5))
       
        # Create label for Target audience
        self.target_label = tk.Label(self.frame, text="Target Audience:")
        self.target_label.grid(row=4, column=0, sticky="e")
        self.target_var = tk.StringVar(self.frame)
        self.target_var.set("")  # Set the initial value to an empty string
        self.target_dropdown = ttk.Combobox(self.frame, textvariable=self.target_var, values=["IT Managers", "Property Managers", "Education Property Managers", "Education Managers", "General Public"], state="readonly")
        self.target_dropdown.grid(row=4, column=1, pady=(0, 5))
 
        # Create a button widget for Submit
        self.submit_button = tk.Button(self.frame, text="Submit", command=self.submit)
        self.submit_button.grid(row=5, columnspan=2, pady=10)
    # Define method called on_validate_booking for validating and checking the booking number entry
    def on_validate_booking(self, P):
        #  Condition to check if the entered number is not zero and not less than or equal to 99999
        if P.isdigit() and int(P) <= 99999 and not P.startswith('0'):
            return True
        elif P == "":
            return True
        else:
            return False
    # Define a method called submit for handling the submission
    def submit(self):
        # Get all entry values
        start_date = self.start_date_entry.get().strip()
        event_date = self.event_date_entry.get().strip()
        current_date = self.current_date_entry.get().strip()
        booking_no = self.booking_entry.get().strip()
        target_audience = self.target_var.get()
   
        # Check if all fields are filled
        if not all([start_date, event_date, current_date, booking_no, target_audience]):
            messagebox.showerror("Error", "Please fill in all fields.")
            self.reset_form()
            return
   
        # Check if all dates are entered in dd/mm/yyyy format
        if not self.validate_date(start_date) or not self.validate_date(event_date) or not self.validate_date(current_date):
            messagebox.showerror("Error", "Please enter dates in dd/mm/yyyy format.")
            self.reset_form()
            return
   
        # Check if the event date and the start date are the same
        if start_date == event_date:
            messagebox.showerror("Error", "Please enter the correct dates.")
            self.reset_form()
            return
   
        # Check if the start date is not before 1990
        if datetime.strptime(start_date, "%d/%m/%Y").year < 1900:
            messagebox.showerror("Error", "Start date should not be before 01/01/1900.")
            self.reset_form()
            return
         # Check if the event date is not before the start date
        if datetime.strptime(event_date, "%d/%m/%Y") <= datetime.strptime(start_date, "%d/%m/%Y"):
            messagebox.showerror("Error", "Event date must be later than start date.")
            self.reset_form()
            return
        # Check if the current date is between the start date and end date
        if datetime.strptime(current_date, "%d/%m/%Y") < datetime.strptime(start_date, "%d/%m/%Y") or datetime.strptime(current_date, "%d/%m/%Y") > datetime.strptime(event_date, "%d/%m/%Y"):
            messagebox.showerror("Error", "Current date must be within start date and event date.")
            self.reset_form()
            return
   
        # Perform prediction
        try:
            booking_no = int(booking_no)
             # Check if the booking number is less than or equal to zero
            if booking_no <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Booking number must be a positive integer.")
            self.reset_form()
            return
        # Check if the type of target audience was chosen
        if not target_audience:
            messagebox.showerror("Error", "Please select a target audience.")
            self.reset_form()
            return
   
        audience_mapping = {"IT Managers": 1, "Property Managers": 2, "Education Property Managers": 3, "Education Managers": 4, "General Public": 5}
        target_audience_code = audience_mapping[target_audience]
   
        # Perform prediction
        try:
            result1, result2 = fm.prediction([start_date, event_date, current_date, booking_no, target_audience_code])
            result3 = fwd.audience_pred(target_audience_code)
            result = [result1, result2, result3]
   
            # Display result
            self.show_result(result)
        except ValueError:
            messagebox.showerror("Error", "Please enter proper dates.")
            self.reset_form()
 
    # Define a method called reset_form to clear all the entries
    def reset_form(self):
        self.start_date_entry.delete(0, tk.END)
        self.event_date_entry.delete(0, tk.END)
        self.current_date_entry.delete(0, tk.END)
        self.booking_entry.delete(0, tk.END)
        self.target_var.set("")
    # Define a method called validate_date to check if the date format is correct
    def validate_date(self, date_string):
        try:
            datetime.strptime(date_string, "%d/%m/%Y")
            return True
        except ValueError:
            return False
    # Define a method called method show_result to display the results of the prediction
    def show_result(self, result):
        total_without_ad, total_with_ad, attendance = result
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Prediction Results")
        result_frame = tk.Frame(self.result_window)
        result_frame.pack(padx=10, pady=10)
   
        result_label = tk.Label(result_frame, text=f"Without any advertisement, the total predicted bookings for the event are: {total_without_ad}\n"
                                                    f"The predicted attendance is: {attendance * 100:.2f}%\n"
                                                    f"So {math.floor(total_without_ad * attendance)} participants are expected to attend.\n\n"
                                                    f"With advertisement, the total predicted bookings for the event are: {total_with_ad}\n"
                                                    f"The predicted attendance is: {attendance * 100:.2f}%\n"
                                                    f"So {math.floor(total_with_ad * attendance)} participants are expected to attend.")
        result_label.pack()
   
        # Bind closing event to the result window
        self.result_window.protocol("WM_DELETE_WINDOW", self.close_app)
       
        # Make the result window modal
        self.result_window.grab_set()
 
    # Define a method called close_app
    def close_app(self):
        # Reset input fields
        self.reset_form()
        # Close the result window
        self.result_window.destroy()
# Define the main function
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
 
if __name__ == "__main__":
    main()
