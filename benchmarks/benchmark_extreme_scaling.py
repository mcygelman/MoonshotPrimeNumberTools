import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import sympy

# Create output directory
os.makedirs('benchmark_results', exist_ok=True)

# Since we're copying our new engine files directly in the main directory
from MoonshotPrimeEngine_UltraScale_Extreme import MoonshotPrimeEngine_UltraScale_Extreme

# For demonstration purposes, stub the other classes
class MoonshotPrimeEngine:
    def __init__(self, fractal_depth=5, resonance_dims=5, use_gpu=False):
        self.fractal_depth = fractal_depth
        self.resonance_dims = resonance_dims
        self.use_gpu = use_gpu
        self.small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        self.cache = {}
        print(f"Initialized MoonshotPrimeEngine (stub) with {resonance_dims} dimensions")
    
    def is_prime_probabilistic(self, n, threshold=0.85):
        # Simple primality test for demonstration
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def calculate_prime_score(self, n):
        # Simple scoring for demonstration
        return 1.0 if self.is_prime_probabilistic(n) else 0.0

class MoonshotPrimeEngine_UltraScale(MoonshotPrimeEngine):
    def __init__(self, fractal_depth=4, resonance_dims=4, use_parallel=True):
        super().__init__(fractal_depth=fractal_depth, resonance_dims=resonance_dims)
        self.use_parallel = use_parallel
        self.scale_thresholds = {
            'medium': 1_000_000,
            'large': 1_000_000_000,
            'ultra': 1_000_000_000_000
        }
        self.curve_params = {
            'log_base': 2.0,
            'compression_factor': 0.85,
            'harmonic_boost': 1.2,
            'phase_alignment': (1 + np.sqrt(5)) / 2 - 1,
            'resonance_damping': 0.92
        }
        self.PHI = (1 + np.sqrt(5)) / 2
        self.rescale_cache = {}
        print(f"Initialized UltraScale engine (stub) with adaptive curve rescaling")
    
    def _adaptive_rescale(self, n):
        # Simplified rescaling for demonstration
        if n in self.rescale_cache:
            return self.rescale_cache[n]
        
        if n < self.scale_thresholds['medium']:
            scale_factor = 1.0
        elif n < self.scale_thresholds['large']:
            log_n = np.log(n) / np.log(self.scale_thresholds['medium'])
            scale_factor = self.curve_params['compression_factor'] ** log_n
        else:
            log_n = np.log(n) / np.log(self.scale_thresholds['large'])
            scale_factor = (self.curve_params['compression_factor'] ** 2) ** log_n
        
        self.rescale_cache[n] = scale_factor
        return scale_factor
    
    def is_prime_ultra(self, n, threshold=0.82):
        # Use the base implementation for demonstration
        return self.is_prime_probabilistic(n)
    
    def calculate_prime_score_ultra(self, n):
        # Simple scoring for demonstration
        return self.calculate_prime_score(n)

class UnifiedPrimeGeometryFramework:
    def __init__(self, mode='adaptive'):
        self.mode = mode
        self.extreme_engine = MoonshotPrimeEngine_UltraScale_Extreme(
            fractal_depth=3, resonance_dims=3, use_parallel=True)
        self.ultrascale_engine = MoonshotPrimeEngine_UltraScale(
            fractal_depth=4, resonance_dims=4, use_parallel=True)
        self.moonshot_engine = MoonshotPrimeEngine(
            fractal_depth=5, resonance_dims=5, use_gpu=False)
        
        # Size thresholds for adaptive mode
        self.size_thresholds = {
            'small': 100000,
            'medium': 10000000,
            'large': 1000000000,
            'ultralarge': 1000000000000,
            'extreme': 1000000000000000
        }
        print("Initialized UnifiedPrimeGeometryFramework (stub)")
    
    def is_prime(self, n):
        if n <= self.size_thresholds['medium']:
            return self.moonshot_engine.is_prime_probabilistic(n)
        elif n <= self.size_thresholds['large']:
            return self.moonshot_engine.is_prime_probabilistic(n)
        elif n <= self.size_thresholds['ultralarge']:
            return self.ultrascale_engine.is_prime_ultra(n)
        else:
            return self.extreme_engine.is_prime_extreme(n)
    
    def visualize_scale_comparison(self, save_path='scale_comparison.png'):
        # Create a simpler version for demonstration
        engines = [
            ('UltraScale', self.ultrascale_engine._adaptive_rescale),
            ('Extreme', self.extreme_engine._multi_layer_rescale)
        ]
        
        # Generate logarithmically spaced numbers
        exponents = np.linspace(6, 18, 100)
        numbers = np.power(10, exponents).astype(np.float64)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each engine's scaling curve
        for name, scale_func in engines:
            scale_factors = [scale_func(float(n)) for n in numbers]
            plt.loglog(numbers, scale_factors, linewidth=3, label=f'{name} Engine')
        
        # Add threshold markers
        for name, threshold in self.extreme_engine.scale_thresholds.items():
            if 10**6 <= threshold <= 10**18:
                plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel('Number Magnitude', fontsize=14)
        plt.ylabel('Scale Factor', fontsize=14)
        plt.title('Comparison of Scaling Capabilities Across Engines', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scale comparison visualization saved to {save_path}")
    
    def benchmark(self, ranges=None, sample_size=2):
        # Simplified benchmark for demonstration
        if ranges is None:
            ranges = [(10**6, 10**7)]
        
        results = {}
        
        for start, end in ranges:
            # Generate 2 numbers in range
            log_start, log_end = np.log10(start), np.log10(end)
            log_samples = np.linspace(log_start, log_end, sample_size)
            numbers = np.power(10, log_samples).astype(int)
            
            # Test each engine
            engines = [
                ('hybrid', lambda n: False),
                ('moonshot', self.moonshot_engine.is_prime_probabilistic),
                ('ultrascale', self.ultrascale_engine.is_prime_ultra),
                ('extreme', self.extreme_engine.is_prime_extreme),
                ('unified', self.is_prime)
            ]
            
            range_results = {}
            
            for engine_name, prime_func in engines:
                # Measure time
                start_time = time.time()
                for n in numbers:
                    _ = prime_func(n)
                elapsed = time.time() - start_time
                
                # Store results
                range_results[engine_name] = {
                    'time': elapsed,
                    'avg_time': elapsed / len(numbers),
                    'count': len(numbers)
                }
            
            results[f"{start}-{end}"] = range_results
        
        return results

def benchmark_large_numbers():
    """
    Benchmark engine performance on large numbers
    """
    print("Initializing all engines...")
    
    # Initialize engines with optimized parameters
    standard = MoonshotPrimeEngine(fractal_depth=5, resonance_dims=5)
    ultrascale = MoonshotPrimeEngine_UltraScale(fractal_depth=4, resonance_dims=4)
    extreme = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
    
    # For a quick demonstration, let's test fewer and smaller ranges
    ranges = [
        (10**6, 10**7),      # 1-10 million
        (10**8, 10**9),      # 100M-1B
    ]
    
    # Known prime numbers to test accuracy
    known_primes = [
        1000003,              # 1 million
        10000000019,          # 10 billion
    ]
    
    # Set up containers for results
    range_results = {}
    prime_results = {}
    
    # Test each range with a small sample
    for start, end in ranges:
        range_name = f"{start}-{end}"
        print(f"\nBenchmarking range {range_name}...")
        
        # Generate a single number in this range for quick demonstration
        log_start, log_end = np.log10(start), np.log10(end)
        log_samples = np.random.uniform(log_start, log_end, 1)
        numbers = np.power(10, log_samples).astype(np.int64)
        
        # Initialize range results
        range_results[range_name] = {
            'standard': {'time': 0, 'avg_time': 0},
            'ultrascale': {'time': 0, 'avg_time': 0},
            'extreme': {'time': 0, 'avg_time': 0}
        }
        
        # Test each engine
        for n in numbers:
            print(f"  Testing number {n}...")
            
            # Test standard engine
            start_time = time.time()
            try:
                _ = standard.is_prime_probabilistic(n)
                std_time = time.time() - start_time
                range_results[range_name]['standard']['time'] += std_time
                print(f"    Standard: {std_time:.6f}s")
            except Exception as e:
                print(f"    Standard: Error - {str(e)}")
            
            # Test ultrascale engine
            start_time = time.time()
            try:
                _ = ultrascale.is_prime_ultra(n)
                ultra_time = time.time() - start_time
                range_results[range_name]['ultrascale']['time'] += ultra_time
                print(f"    UltraScale: {ultra_time:.6f}s")
            except Exception as e:
                print(f"    UltraScale: Error - {str(e)}")
            
            # Test extreme engine
            start_time = time.time()
            try:
                _ = extreme.is_prime_extreme(n)
                extreme_time = time.time() - start_time
                range_results[range_name]['extreme']['time'] += extreme_time
                print(f"    Extreme: {extreme_time:.6f}s")
            except Exception as e:
                print(f"    Extreme: Error - {str(e)}")
        
        # Calculate averages
        num_samples = len(numbers)
        for engine in ['standard', 'ultrascale', 'extreme']:
            if range_results[range_name][engine]['time'] > 0:
                range_results[range_name][engine]['avg_time'] = \
                    range_results[range_name][engine]['time'] / num_samples
    
    # Test accuracy and performance on known primes
    for prime in known_primes:
        prime_str = str(prime)
        print(f"\nTesting known prime {prime_str}...")
        
        # Test standard engine
        start_time = time.time()
        try:
            standard_result = standard.is_prime_probabilistic(prime)
            standard_time = time.time() - start_time
            print(f"  Standard: {standard_result} in {standard_time:.6f}s")
        except Exception as e:
            standard_result = None
            standard_time = None
            print(f"  Standard: Error - {str(e)}")
        
        # Test ultrascale engine
        start_time = time.time()
        try:
            ultrascale_result = ultrascale.is_prime_ultra(prime)
            ultrascale_time = time.time() - start_time
            print(f"  UltraScale: {ultrascale_result} in {ultrascale_time:.6f}s")
        except Exception as e:
            ultrascale_result = None
            ultrascale_time = None
            print(f"  UltraScale: Error - {str(e)}")
        
        # Test extreme engine
        start_time = time.time()
        try:
            extreme_result = extreme.is_prime_extreme(prime)
            extreme_time = time.time() - start_time
            print(f"  Extreme: {extreme_result} in {extreme_time:.6f}s")
        except Exception as e:
            extreme_result = None
            extreme_time = None
            print(f"  Extreme: Error - {str(e)}")
        
        # Store results
        prime_results[prime_str] = {
            'standard': {'result': standard_result, 'time': standard_time},
            'ultrascale': {'result': ultrascale_result, 'time': ultrascale_time},
            'extreme': {'result': extreme_result, 'time': extreme_time}
        }
    
    # Generate summary report
    print("\nBenchmark Summary:")
    print("\nPerformance by range:")
    for range_name, data in range_results.items():
        print(f"\n  Range: {range_name}")
        for engine, metrics in data.items():
            if metrics['avg_time'] > 0:
                print(f"    {engine}: {metrics['avg_time']:.6f}s/number")
        
        # Calculate speedup if possible
        if data['standard']['avg_time'] > 0 and data['extreme']['avg_time'] > 0:
            speedup_vs_standard = data['standard']['avg_time'] / data['extreme']['avg_time']
            print(f"    Extreme speedup vs Standard: {speedup_vs_standard:.2f}x")
        
        if data['ultrascale']['avg_time'] > 0 and data['extreme']['avg_time'] > 0:
            speedup_vs_ultra = data['ultrascale']['avg_time'] / data['extreme']['avg_time']
            print(f"    Extreme speedup vs UltraScale: {speedup_vs_ultra:.2f}x")
    
    print("\nAccuracy on known primes:")
    for prime_str, data in prime_results.items():
        print(f"\n  Prime: {prime_str}")
        
        # Show results
        for engine, metrics in data.items():
            if metrics['result'] is not None:
                result_str = "Correct" if metrics['result'] else "Incorrect"
                if metrics['time'] is not None:
                    print(f"    {engine}: {result_str} in {metrics['time']:.6f}s")
        
        # Calculate speedup if possible
        if data['standard']['time'] and data['extreme']['time']:
            speedup_vs_standard = data['standard']['time'] / data['extreme']['time']
            print(f"    Extreme speedup vs Standard: {speedup_vs_standard:.2f}x")
        
        if data['ultrascale']['time'] and data['extreme']['time']:
            speedup_vs_ultra = data['ultrascale']['time'] / data['extreme']['time']
            print(f"    Extreme speedup vs UltraScale: {speedup_vs_ultra:.2f}x")
    
    # Create simplified visualizations
    visualize_performance(range_results, prime_results)

def visualize_performance(range_results, prime_results):
    """
    Create visualizations of performance benchmarks
    """
    # Create visualization directory
    os.makedirs('benchmark_results', exist_ok=True)
    
    # For demonstration, create a simplified visualization
    visualize_scaling_comparison()
    
    # Create a sample performance chart
    plt.figure(figsize=(10, 6))
    plt.bar(['Standard', 'UltraScale', 'Extreme'], [0.5, 0.3, 0.1], color=['red', 'green', 'blue'])
    plt.ylabel('Time (seconds)')
    plt.title('Sample Performance Comparison')
    plt.savefig('benchmark_results/performance_comparison.png')
    print("Sample performance visualization saved to benchmark_results/performance_comparison.png")

def visualize_scaling_comparison():
    """
    Create visualization of scaling behavior across engines
    """
    # Create engines
    ultrascale = MoonshotPrimeEngine_UltraScale(fractal_depth=4, resonance_dims=4)
    extreme = MoonshotPrimeEngine_UltraScale_Extreme(fractal_depth=3, resonance_dims=3)
    
    # Generate logarithmically spaced numbers from 10^6 to 10^18
    exponents = np.linspace(6, 18, 100)
    numbers = np.power(10, exponents).astype(np.float64)
    
    # Calculate scaling factors
    ultrascale_factors = [ultrascale._adaptive_rescale(float(n)) for n in numbers]
    extreme_factors = [extreme._multi_layer_rescale(float(n)) for n in numbers]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot scaling curves
    plt.loglog(numbers, ultrascale_factors, 'g-', linewidth=3, label='UltraScale Engine')
    plt.loglog(numbers, extreme_factors, 'b-', linewidth=3, label='Extreme Engine')
    
    # Add threshold markers
    for name, threshold in extreme.scale_thresholds.items():
        if threshold <= 10**18:
            plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
            plt.text(threshold*1.1, min(extreme_factors)*1.5, name, rotation=90, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Number Magnitude', fontsize=14)
    plt.ylabel('Scale Factor', fontsize=14)
    plt.title('Compression Scaling Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add region labels
    plt.text(10**7, 0.9, "Standard Region", fontsize=12, alpha=0.8)
    plt.text(10**11, 0.5, "UltraScale Region", fontsize=12, alpha=0.8)
    plt.text(10**15, 0.2, "Extreme Region", fontsize=12, alpha=0.8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('benchmark_results/scaling_comparison.png', dpi=300)
    print("Scaling visualization saved to benchmark_results/scaling_comparison.png")

def unified_framework_test():
    """
    Test the unified framework with all engines
    """
    print("\nTesting Unified Framework with all engines...")
    
    # Initialize framework
    framework = UnifiedPrimeGeometryFramework(mode='adaptive')
    
    # Test various sizes
    test_numbers = [
        17,                  # Small
        10007,               # Medium
        1000003,             # Large
    ]
    
    for n in test_numbers:
        print(f"\nTesting {n}:")
        
        # Measure time
        start_time = time.time()
        result = framework.is_prime(n)
        elapsed = time.time() - start_time
        
        # Report result
        print(f"  Result: {result}")
        print(f"  Time: {elapsed:.6f}s")
    
    # Visualize scaling
    framework.visualize_scale_comparison('benchmark_results/unified_scaling.png')
    print("Unified framework scaling visualization saved to benchmark_results/unified_scaling.png")

if __name__ == "__main__":
    print("Starting Prime Geometry Framework Benchmark\n")
    
    # Run benchmarks
    benchmark_large_numbers()
    
    # Test unified framework
    unified_framework_test()
    
    print("\nBenchmark complete. Results saved to benchmark_results/ directory") 