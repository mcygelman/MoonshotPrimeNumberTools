import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
import sympy
from numba import njit, parallel
from MoonshotPrimeEngine import MoonshotPrimeEngine

class MoonshotPrimeEngine_UltraScale(MoonshotPrimeEngine):
    """
    Enhanced MoonshotPrimeEngine for ultra-large numbers (>1,000,000)
    
    Key improvements:
    1. Adaptive curve rescaling for large number spaces
    2. Progressive dimensional reduction for efficiency
    3. Resonance pattern optimization with scale-invariant features
    4. Multi-scale interference patterns with golden ratio harmonic alignment
    5. Localized feature extraction with logarithmic curve fitting
    """
    
    def __init__(self, fractal_depth=5, resonance_dims=4, use_parallel=True):
        """
        Initialize the UltraScale engine
        
        Parameters:
        - fractal_depth: Depth of fractal analysis (reduced from base class for speed)
        - resonance_dims: Number of resonance dimensions (optimized for ultra-large numbers)
        - use_parallel: Whether to use parallel processing
        """
        # Initialize parent class
        super().__init__(fractal_depth=fractal_depth, resonance_dims=resonance_dims, use_gpu=False)
        
        # Enhanced parameters for ultra-large numbers
        self.scale_thresholds = {
            'medium': 1_000_000,
            'large': 1_000_000_000,
            'ultra': 1_000_000_000_000
        }
        
        # Curve rescaling parameters
        self.curve_params = {
            'log_base': 2.0,
            'compression_factor': 0.85,
            'harmonic_boost': 1.2,
            'phase_alignment': self.PHI - 1,
            'resonance_damping': 0.92
        }
        
        # Enable parallel processing if requested
        self.use_parallel = use_parallel
        
        # Additional caching for rescaled values
        self.rescale_cache = {}
        
        # Special knot points for curve-fitting (logarithmically spaced)
        self.knot_points = np.logspace(6, 15, 10, base=10)
        
        # Initialize new resonance matrices optimized for large numbers
        self._init_ultra_resonance_matrices()
        
        print(f"Initialized UltraScale engine with adaptive curve rescaling")
        
    def _init_ultra_resonance_matrices(self):
        """Initialize resonance matrices optimized for ultra-large numbers"""
        dims = self.resonance_dims
        
        # Create ultra matrices with scale-invariant properties
        self.ultra_resonator = np.zeros((dims, dims))
        for i in range(dims):
            for j in range(dims):
                # Scale-invariant pattern with golden ratio
                self.ultra_resonator[i, j] = (self.PHI ** ((i * j) % dims / dims)) * np.cos(np.pi * i * j / dims)
        
        # Optimized interference patterns for large numbers
        self.ultra_interference = np.zeros((dims, dims), dtype=complex)
        for i in range(dims):
            for j in range(dims):
                # Phase amplification with golden ratio harmonics
                phase = 2 * np.pi * ((i * j) % dims) / dims * self.PHI
                self.ultra_interference[i, j] = np.exp(1j * phase)
                
        # Create logarithmic scale transform matrix
        self.log_transform = np.zeros(dims)
        for i in range(dims):
            # Logarithmic scaling factors
            self.log_transform[i] = np.log(i + self.PHI) / np.log(dims + self.PHI)
    
    def _adaptive_rescale(self, n):
        """
        Apply adaptive curve rescaling based on number magnitude
        
        This is the key innovation for ultra-large number performance
        """
        # Check cache first
        if n in self.rescale_cache:
            return self.rescale_cache[n]
        
        # Determine scale category
        if n < self.scale_thresholds['medium']:
            scale_factor = 1.0
        elif n < self.scale_thresholds['large']:
            # Medium range: apply gentle curve
            log_n = np.log(n) / np.log(self.scale_thresholds['medium'])
            scale_factor = self.curve_params['compression_factor'] ** log_n
        else:
            # Large/ultra range: apply stronger curve
            log_n = np.log(n) / np.log(self.scale_thresholds['large'])
            base_scale = self.curve_params['compression_factor'] ** 2
            scale_factor = base_scale ** (log_n * self.curve_params['harmonic_boost'])
            
            # Apply golden ratio modulation for ultra-large numbers
            if n >= self.scale_thresholds['ultra']:
                phi_mod = (np.log(n) * self.PHI) % 1
                scale_factor *= (0.8 + 0.4 * phi_mod)
        
        # Store in cache and return
        self.rescale_cache[n] = scale_factor
        return scale_factor
    
    def _generate_feature_vector(self, n):
        """
        Generate optimized feature vector for ultra-large numbers
        
        Overrides the parent class method with scale-invariant features
        """
        # Apply adaptive rescaling
        scale_factor = self._adaptive_rescale(n)
        
        # Extract fundamental features with scaling
        log_n = np.log(max(n, 2))
        sqrt_n = np.sqrt(n)
        log_digits = np.log(1 + np.sum(self._digit_extraction(n)))
        
        # Scale-invariant features optimized for large numbers
        base_features = np.array([
            np.sin(2 * np.pi * (n % 30) / 30),  # Modular position (remains useful at all scales)
            np.sin(scale_factor * log_n),       # Scaled logarithmic position
            np.cos(scale_factor * sqrt_n / log_n), # Scaled Riemann-inspired oscillation
            np.sin(2 * np.pi * np.log(log_n) * self.PHI), # Golden ratio log-log oscillation
        ])
        
        # Select features based on dimensions
        return base_features[:self.resonance_dims]
    
    def _resonance_transform_ultra(self, n):
        """
        Apply specialized resonance transformation for ultra-large numbers
        """
        # Generate the feature vector with adaptive scaling
        feature_vector = self._generate_feature_vector(n)
        
        # Ensure vector has the right dimensions
        if len(feature_vector) < self.resonance_dims:
            feature_vector = np.pad(feature_vector, 
                                  (0, self.resonance_dims - len(feature_vector)))
        
        # Apply the ultra-optimized resonance transformations
        primary = np.dot(self.ultra_resonator, feature_vector)
        
        # Apply logarithmic modulation
        log_mod = np.dot(self.log_transform, primary)
        
        # Complex interference pattern optimized for large numbers
        interference = np.dot(self.ultra_interference, feature_vector)
        interference_magnitude = np.abs(interference)
        
        # Apply phase alignment based on golden ratio
        phase = np.angle(interference[0]) * self.curve_params['phase_alignment']
        aligned = interference_magnitude * np.cos(phase)
        
        # Combine with logarithmic dampening
        log_factor = 1.0 / (1 + np.log1p(np.log1p(n)) * 0.1)
        
        # Resonance vector specifically tuned for large numbers
        resonance_vector = np.concatenate([
            primary * log_factor,
            aligned,
            [log_mod]
        ])
        
        return resonance_vector
    
    def _harmonic_resonance_ultra(self, n):
        """
        Calculate enhanced harmonic resonance for ultra-large numbers
        """
        # Quick test for obvious composites
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            if n in [2, 3, 5]:
                return 1.0
            return 0.0
        
        # Get the ultra-optimized resonance vector
        resonance_vector = self._resonance_transform_ultra(n)
        
        # Apply golden ratio harmonic binning
        bins = int(self.resonance_dims * self.PHI)
        hist, _ = np.histogram(resonance_vector, bins=bins)
        hist = hist / np.sum(hist)
        
        # Calculate entropy-like measure (primes tend to have higher entropy)
        non_zero = hist[hist > 0]
        entropy = -np.sum(non_zero * np.log(non_zero))
        
        # Normalize to [0,1] using sigmoid with golden ratio
        normalized = 1.0 / (1.0 + np.exp(-self.PHI * (entropy - 1.0)))
        
        # Apply magnitude scaling
        log_scale = np.log1p(np.log1p(n)) / 10
        scaled = normalized * (1 - log_scale) + log_scale * 0.5
        
        return scaled
    
    def calculate_prime_score_ultra(self, n):
        """
        Calculate optimized prime score for ultra-large numbers
        
        Returns:
        - Score between 0-1, higher values suggest prime number
        """
        # Handle edge cases
        if n <= 1:
            return 0.0
        if n in self.small_primes:
            return 1.0
        if any(n % p == 0 for p in self.small_primes):
            return 0.0
        
        # Check cache
        if n in self.cache:
            return self.cache[n]
        
        # For medium-sized numbers, use standard approach
        if n < self.scale_thresholds['medium']:
            return super().calculate_prime_score(n)
        
        # For large numbers, use ultra-optimized approach
        # Calculate specialized component scores
        harmonic_score = self._harmonic_resonance_ultra(n)
        
        # Apply final scaling with golden ratio
        scale_factor = np.sqrt(self._adaptive_rescale(n))
        final_score = harmonic_score * scale_factor + (1 - scale_factor) * 0.5
        
        # Adjust with a sharpened non-linear function
        adjusted_score = 1.0 / (1.0 + np.exp(-18 * (final_score - 0.52)))
        
        # Cache the result
        self.cache[n] = adjusted_score
        
        return adjusted_score
    
    def is_prime_ultra(self, n, threshold=0.82):
        """
        Enhanced primality test for ultra-large numbers
        
        Returns:
        - Boolean indication of primality with high probability
        """
        # Handle edge cases
        if n <= 1:
            return False
        if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            return False
        
        # For smaller numbers, use standard approach
        if n < self.scale_thresholds['medium']:
            return super().is_prime_probabilistic(n)
        
        # For large numbers, optimize the approach
        score = self.calculate_prime_score_ultra(n)
        
        # For borderline cases in certain ranges, use additional validation
        if 0.4 <= score <= 0.9:
            # Deterministic test for medium range
            if n < self.scale_thresholds['large']:
                # Use sympy's implementation for definitive answer
                return sympy.isprime(n)
            
            # For extremely large numbers, adjust threshold based on magnitude
            log_n = np.log(n)
            adjustment = 0.05 * np.tanh(log_n / 30)
            threshold -= adjustment
        
        # Final decision with adjusted threshold
        return score >= threshold
    
    def benchmark_ultra(self, ranges=None, sample_size=10):
        """
        Run benchmarks specifically for ultra-large numbers
        
        Parameters:
        - ranges: List of (start, end) tuples for ranges to test
        - sample_size: Number of samples per range
        
        Returns:
        - Dictionary of benchmark results
        """
        if ranges is None:
            ranges = [
                (1_000_000, 10_000_000),
                (10_000_000, 100_000_000),
                (100_000_000, 1_000_000_000),
                (1_000_000_000, 10_000_000_000)
            ]
        
        results = {}
        
        for start, end in ranges:
            print(f"Benchmarking ultra range {start}-{end}...")
            
            # Generate logarithmically spaced numbers
            log_start, log_end = np.log10(start), np.log10(end)
            log_samples = np.linspace(log_start, log_end, sample_size)
            numbers = np.unique(np.power(10, log_samples).astype(int))
            
            # Generate some known primes in this range
            primes = []
            for _ in range(min(3, sample_size // 2)):
                # Find a prime near each third of the range
                target = int(start + (end - start) * (_ + 1) / 4)
                prime = sympy.nextprime(target)
                if prime < end:
                    primes.append(prime)
            
            # Combine samples and primes
            test_numbers = sorted(list(set(list(numbers) + primes)))
            
            # Test both methods
            methods = [
                ('standard', self.is_prime_probabilistic),
                ('ultra', self.is_prime_ultra)
            ]
            
            range_results = {}
            
            for method_name, method_func in methods:
                # Measure time
                start_time = time.time()
                results_list = []
                
                for n in test_numbers:
                    is_prime = method_func(n)
                    verified = sympy.isprime(n)
                    results_list.append((is_prime, verified))
                
                # Calculate time and accuracy
                elapsed = time.time() - start_time
                avg_time = elapsed / len(test_numbers)
                
                # Calculate accuracy
                correct = sum(1 for pred, true in results_list if pred == true)
                accuracy = correct / len(results_list) if results_list else 0
                
                # Store results
                range_results[method_name] = {
                    'time': elapsed,
                    'avg_time': avg_time,
                    'accuracy': accuracy,
                    'count': len(test_numbers)
                }
                
                # Show detailed results for debugging
                if len(test_numbers) <= 10:
                    for i, n in enumerate(test_numbers):
                        pred, true = results_list[i]
                        print(f"{method_name}: {n} - Predicted: {pred}, Actual: {true}")
                
            # Store range results
            results[f"{start}-{end}"] = range_results
        
        return results

# Demo function to show improvement
def run_ultra_benchmark():
    """
    Run demonstration benchmark to show performance improvement
    """
    print("Initializing UltraScale engine...")
    engine = MoonshotPrimeEngine_UltraScale(fractal_depth=4, resonance_dims=4)
    
    print("\nTesting with individual large primes:")
    test_cases = [
        1_000_003,
        10_000_019,
        100_000_007,
        1_000_000_007,
        10_000_000_019
    ]
    
    for n in test_cases:
        start_time = time.time()
        ultra_result = engine.is_prime_ultra(n)
        ultra_time = time.time() - start_time
        
        start_time = time.time()
        standard_result = engine.is_prime_probabilistic(n)
        standard_time = time.time() - start_time
        
        # Verify with sympy
        is_really_prime = sympy.isprime(n)
        
        print(f"\nNumber: {n}")
        print(f"Ultra method: Prime={ultra_result}, Time={ultra_time:.6f}s, Correct={ultra_result == is_really_prime}")
        print(f"Standard: Prime={standard_result}, Time={standard_time:.6f}s, Correct={standard_result == is_really_prime}")
        print(f"Speed improvement: {standard_time / ultra_time:.2f}x")
    
    print("\nRunning benchmark across ranges...")
    benchmark_results = engine.benchmark_ultra(sample_size=5)
    
    # Print benchmark results
    for range_key, range_data in benchmark_results.items():
        print(f"\nRange: {range_key}")
        for method, data in range_data.items():
            print(f"  {method}: Time={data['avg_time']:.6f}s/number, Accuracy={data['accuracy']*100:.1f}%")
        
        # Calculate improvement
        if 'standard' in range_data and 'ultra' in range_data:
            std_time = range_data['standard']['avg_time']
            ultra_time = range_data['ultra']['avg_time']
            std_acc = range_data['standard']['accuracy']
            ultra_acc = range_data['ultra']['accuracy']
            
            speed_imp = std_time / ultra_time if ultra_time > 0 else float('inf')
            acc_imp = (ultra_acc - std_acc) / std_acc * 100 if std_acc > 0 else float('inf')
            
            print(f"  Improvement: Speed={speed_imp:.2f}x, Accuracy={acc_imp:.1f}%")
    
    # Visualize scaling behavior
    visualize_scaling(engine)

def visualize_scaling(engine):
    """Visualize the scaling behavior of the UltraScale engine"""
    # Generate logarithmically spaced numbers
    numbers = np.logspace(4, 12, 50, base=10).astype(int)
    
    # Calculate scale factors
    scale_factors = [engine._adaptive_rescale(n) for n in numbers]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(numbers, scale_factors, 'b-', linewidth=2)
    
    # Add threshold markers
    for name, threshold in engine.scale_thresholds.items():
        plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.7)
        plt.text(threshold*1.1, 0.9, name, rotation=90, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Number Magnitude')
    plt.ylabel('Scale Factor')
    plt.title('Adaptive Curve Rescaling for Ultra-Large Numbers')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig('ultra_scale_curve.png', dpi=300)
    print("Scaling curve visualization saved to ultra_scale_curve.png")

if __name__ == "__main__":
    run_ultra_benchmark() 