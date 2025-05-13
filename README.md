# Moonshot Prime Engine - Raspberry Pi Edition

Welcome to the Moonshot Prime Engine, an experimental Python toolkit for detecting primality of ultra-large numbers, now packaged for easy use on your Raspberry Pi!

This engine utilizes a novel geometric approach to primality testing, offering unique performance characteristics, especially for resource-constrained devices.

## Why is it Special?

Traditional primality testing algorithms often become very slow or consume significant memory when dealing with extremely large numbers. The Moonshot Prime Engine is designed to address these challenges:

- **Scale-Invariant Performance:** The flagship `MoonshotPrimeEngine_UltraScale_Extreme` aims to maintain consistent performance even as numbers grow to trillions, quadrillions, and beyond.
- **Low Memory Footprint:** Optimized to use fewer resources than many standard libraries, making it suitable for devices with limited RAM like the Raspberry Pi.
- **Innovative Approach:** Explores geometric patterns and resonance structures rather than purely traditional number-theoretic methods.
- **High-Precision Handling:** Includes mechanisms to accurately perform calculations on numbers larger than standard floating-point precision can handle.

## Why Use it on Raspberry Pi?

Raspberry Pi users can particularly benefit from the Moonshot Prime Engine:

- **Efficient Primality Testing:** Perform primality tests on very large numbers without overwhelming your Pi's resources.
- **Cryptography & Security Projects:** Useful for projects involving key generation or other cryptographic operations that require large primes.
- **Mathematical Exploration:** Explore number theory and prime distributions on an accessible platform.
- **Educational Tool:** Learn about novel approaches to computational number theory.
- **Battery-Friendly:** Its potential for lower CPU utilization compared to some brute-force methods can be beneficial for battery-powered Pi projects.

## Installation

We've aimed to make installation simple.

**Prerequisites:**

- A Raspberry Pi (any model should work, but performance will vary).
- Python 3.6 or newer.
- `pip` (Python's package installer).

**Steps:**

1.  **Ensure System Dependencies for GMPY2:**
    `gmpy2` is a powerful library that this engine uses. It might require some system libraries to be installed first. On Raspberry Pi OS (and other Debian-based systems), you might need:

    ```bash
    sudo apt-get update
    sudo apt-get install libgmp-dev libmpfr-dev libmpc-dev
    ```

2.  **Install the Package:**
    Once you have the package files (e.g., downloaded from a release or cloned from a repository):
    Navigate to the directory containing this `README.md` and the `setup.py` file in your terminal.
    Then run:
    ```bash
    pip install .
    ```
    This will install the `moonshot_prime` package and its Python dependencies (`numpy`, `scipy`, `matplotlib`, `sympy`, `numba`, `gmpy2`).

## How to Use It

Here's a basic example of how to use the engine in your Python scripts:

````python
from moonshot_prime import MoonshotPrimeEngine_UltraScale_Extreme

# Initialize the engine (this might take a few seconds the first time)
# For Raspberry Pi, using lower fractal_depth might be advisable for quicker init/runs
# if performance is critical for very large numbers.
# Default is fractal_depth=4, resonance_dims=3
try:
    engine = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
    print("Moonshot Prime Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing engine: {e}")
    print("Please ensure all dependencies (numpy, scipy, numba, gmpy2, etc.) are installed correctly.")
    exit()

# Number to test (can be a very large integer)
# Example: A 100-digit number
number_to_test = 10**99 + 7 # A known prime
# number_to_test = 4455776455287638884938744333256765434343232443

print(f"\nTesting number: {number_to_test}")

# Test if the number is prime
try:
    is_it_prime = engine.is_prime_extreme(number_to_test)
    print(f"Is the number likely prime? {is_it_prime}")
except Exception as e:
    print(f"Error during primality test: {e}")

# Calculate the primality score (a value between 0 and 1)
try:
    score = engine.calculate_prime_score_extreme(number_to_test)
    print(f"Primality score: {score:.6f}")
except Exception as e:
    print(f"Error calculating score: {e}")

# Find the next prime after a number
try:
    start_num = 10**10 # Example: find next prime after 10 billion
    if number_to_test > 10**12: # For very large numbers, next_prime can be slow.
        start_num = number_to_test // (10**6) # Example: find next prime near a smaller number
        print(f"(For demonstration, finding next prime after a smaller number: {start_num})")

    next_p = engine.next_prime(start_num)
    print(f"The next prime after {start_num} is: {next_p}")
except Exception as e:
    print(f"Error finding next prime: {e}")

## GUI Example

This project includes a self-contained graphical user interface (GUI) example in the `examples/GUI_Original` directory. It demonstrates the usage of the prime engine visually.

To run the GUI:

1.  **Navigate to the GUI directory:**
    ```bash
    cd examples/GUI_Original
    ```
2.  **Install GUI-specific dependencies:**
    The GUI has its own set of requirements. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the GUI:**
    ```bash
    python prime_gui.py
    ```

Note: This GUI example uses a copy of the engine code (`MoonshotPrimeEngine_UltraScale_Extreme.py`) for simplicity and may not reflect the absolute latest changes in the main library.

## License

This software is provided under a custom license.

- **Free for Non-Commercial Use:** You can use, modify, and distribute it freely for personal projects, academic research, and educational purposes.
- **Commercial Use Requires License:** If you intend to use this software for commercial purposes, please contact us at `AetheCoreContact@gamil.com` to arrange a commercial license.

Please see the `LICENSE` file for full details. (Patent Pending: US 63/802,543)

## Contributing & Feedback

This is an experimental engine. Feedback, bug reports, and contributions are welcome! Please contact `AetheCoreContact@gamil.com`.
````
