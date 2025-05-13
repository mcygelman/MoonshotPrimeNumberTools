import tkinter as tk
from tkinter import scrolledtext, messagebox, font
import time
import threading

# Try importing the engine
try:
    from MoonshotPrimeEngine_UltraScale_Extreme import MoonshotPrimeEngine_UltraScale_Extreme
except ImportError as e:
    messagebox.showerror("Import Error", 
                         f"Failed to import MoonshotPrimeEngine_UltraScale_Extreme:\n{e}\n\n" 
                         f"Ensure the engine file is in the same directory and all dependencies "
                         f"(numpy, scipy, numba, gmpy2, etc.) are installed.")
    exit()

class PrimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Moonshot Prime Engine GUI")
        self.root.geometry("600x450")

        # Set modern font
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Segoe UI", size=10)
        self.bold_font = font.Font(family="Segoe UI", size=10, weight="bold")
        self.title_font = font.Font(family="Segoe UI", size=12, weight="bold")

        # --- Status Bar (define before engine init) ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready") # Initial state before engine init
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Engine Initialization ---
        self.engine = None
        self.init_engine()

        # --- GUI Layout ---
        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(main_frame, text="Moonshot Prime Tester", font=self.title_font)
        title_label.pack(pady=(0, 10))

        # Input Frame
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)

        input_label = tk.Label(input_frame, text="Number to Test:", font=self.bold_font)
        input_label.pack(side=tk.LEFT, padx=(0, 5))

        self.number_entry = tk.Entry(input_frame, width=40, font=self.default_font)
        self.number_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Add placeholder text behavior
        self.add_placeholder(self.number_entry, "Enter a large integer...")


        # Button Frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.test_button = tk.Button(button_frame, text="Is Prime?", command=self.run_is_prime, width=15, font=self.default_font)
        self.test_button.pack(side=tk.LEFT, padx=5)

        self.score_button = tk.Button(button_frame, text="Get Score", command=self.run_get_score, width=15, font=self.default_font)
        self.score_button.pack(side=tk.LEFT, padx=5)

        self.next_prime_button = tk.Button(button_frame, text="Find Next Prime", command=self.run_find_next_prime, width=15, font=self.default_font)
        self.next_prime_button.pack(side=tk.LEFT, padx=5)

        # Results Frame
        results_frame = tk.Frame(main_frame, bd=1, relief=tk.SUNKEN)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=10, width=70, state=tk.DISABLED, font=self.default_font)
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    def add_placeholder(self, entry, placeholder):
        entry.insert(0, placeholder)
        entry.config(fg='grey')
        entry.bind("<FocusIn>", lambda event: self.on_entry_click(event, entry, placeholder))
        entry.bind("<FocusOut>", lambda event: self.on_focusout(event, entry, placeholder))

    def on_entry_click(self, event, entry, placeholder):
        if entry.get() == placeholder:
           entry.delete(0, "end") # delete all the text in the entry
           entry.insert(0, '') #Insert blank for user input
           entry.config(fg='black')

    def on_focusout(self, event, entry, placeholder):
        if entry.get() == '':
            entry.insert(0, placeholder)
            entry.config(fg='grey')

    def init_engine(self):
        self.status_var.set("Initializing engine...")
        self.root.update_idletasks()
        try:
            self.engine = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
            self.status_var.set("Engine initialized. Ready.")
        except Exception as e:
            messagebox.showerror("Engine Initialization Error", f"Could not initialize the engine:\n{e}")
            self.status_var.set("Engine initialization failed!")
            self.engine = None # Ensure engine is None if init fails

    def log_result(self, message):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END) # Scroll to the end
        self.results_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def run_test(self, test_function):
        if not self.engine:
            messagebox.showerror("Error", "Engine not initialized.")
            return

        number_str = self.number_entry.get()
        if number_str == "Enter a large integer..." or not number_str:
            messagebox.showwarning("Input Required", "Please enter a number to test.")
            return

        try:
            n = int(number_str)
            if n <= 1:
                 messagebox.showwarning("Invalid Input", "Please enter an integer greater than 1.")
                 return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer.")
            return

        self.log_result(f"--- Testing {n} ---")
        self.status_var.set(f"Testing {n}...")
        self.test_button.config(state=tk.DISABLED)
        self.score_button.config(state=tk.DISABLED)
        self.next_prime_button.config(state=tk.DISABLED)
        self.root.update_idletasks()

        start_time = time.time()
        
        # Run the engine test in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=test_function, args=(n, start_time), daemon=True)
        thread.start()

    def execute_is_prime(self, n, start_time):
        try:
            result = self.engine.is_prime_extreme(n)
            elapsed = time.time() - start_time
            self.log_result(f"Result: {'Likely Prime' if result else 'Composite'}")
            self.log_result(f"Time Taken: {elapsed:.4f} seconds")
            self.status_var.set(f"Test complete for {n}.")
        except Exception as e:
            self.log_result(f"Error during testing: {e}")
            self.status_var.set(f"Error testing {n}.")
            messagebox.showerror("Engine Error", f"An error occurred during testing:\n{e}")
        finally:
            self.test_button.config(state=tk.NORMAL)
            self.score_button.config(state=tk.NORMAL)
            self.next_prime_button.config(state=tk.NORMAL)

    def execute_get_score(self, n, start_time):
        try:
            score = self.engine.calculate_prime_score_extreme(n)
            elapsed = time.time() - start_time
            # Determine primality based on score and band
            low_amb, high_amb = self.engine.ambivalent_band
            threshold = 0.80 # Default threshold from engine, adjust if needed
            if score >= threshold:
                verdict = "Likely Prime"
            elif score < low_amb:
                verdict = "Composite"
            elif score >= low_amb and score < high_amb:
                 verdict = "Ambiguous (needs Miller-Rabin)"
            else: # Between high_amb and threshold
                verdict = "Likely Composite"
                
            self.log_result(f"Primality Score: {score:.6f}")
            self.log_result(f"Verdict based on score: {verdict}")
            self.log_result(f"Time Taken: {elapsed:.4f} seconds")
            self.status_var.set(f"Score calculation complete for {n}.")
        except Exception as e:
            self.log_result(f"Error calculating score: {e}")
            self.status_var.set(f"Error calculating score for {n}.")
            messagebox.showerror("Engine Error", f"An error occurred calculating score:\n{e}")
        finally:
            self.test_button.config(state=tk.NORMAL)
            self.score_button.config(state=tk.NORMAL)
            self.next_prime_button.config(state=tk.NORMAL)

    def run_is_prime(self):
        self.run_test(self.execute_is_prime)

    def run_get_score(self):
        self.run_test(self.execute_get_score)

    # --- Next Prime Functionality ---
    def execute_find_next_prime(self, n, start_time):
        try:
            next_p = self.engine.next_prime(n)
            elapsed = time.time() - start_time
            self.log_result(f"Next prime after {n} is: {next_p}")
            self.log_result(f"Time Taken: {elapsed:.4f} seconds")
            self.status_var.set(f"Found next prime after {n}.")
        except Exception as e:
            self.log_result(f"Error finding next prime: {e}")
            self.status_var.set(f"Error finding next prime after {n}.")
            messagebox.showerror("Engine Error", f"An error occurred finding next prime:\n{e}")
        finally:
            self.test_button.config(state=tk.NORMAL)
            self.score_button.config(state=tk.NORMAL)
            self.next_prime_button.config(state=tk.NORMAL)

    def run_find_next_prime(self):
        if not self.engine:
            messagebox.showerror("Error", "Engine not initialized.")
            return

        number_str = self.number_entry.get()
        start_n = 1 # Default to find prime after 1 (which is 2) if input is empty/invalid
        
        if number_str and number_str != "Enter a large integer...":
            try:
                start_n = int(number_str)
                if start_n < 0:
                    messagebox.showwarning("Invalid Input", "Please enter a non-negative integer.")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer or leave blank.")
                return
        else:
             self.log_result("Input empty, finding first prime...")

        self.log_result(f"--- Finding next prime after {start_n} ---")
        self.status_var.set(f"Finding next prime after {start_n}...")
        self.test_button.config(state=tk.DISABLED)
        self.score_button.config(state=tk.DISABLED)
        self.next_prime_button.config(state=tk.DISABLED)
        self.root.update_idletasks()

        start_time = time.time()
        
        # Run the engine test in a separate thread
        thread = threading.Thread(target=self.execute_find_next_prime, args=(start_n, start_time), daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PrimeGUI(root)
    root.mainloop() 