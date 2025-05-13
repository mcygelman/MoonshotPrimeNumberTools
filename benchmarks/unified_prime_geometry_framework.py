import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.interpolate import BSpline, splrep
import scipy.ndimage

# Try to import acceleration libraries
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Create pass-through decorator
    def njit(func):
        return func
    
    def prange(x):
        return range(x)
    
    NUMBA_AVAILABLE = False
    print("Numba not available. Using slower Python implementation.")

class UnifiedPrimeGeometryFramework:
    """
    Unified Prime Geometry Framework
    
    Integrates the discoveries from HybridPrimeEngineV4.0, MoonshotPrimeEngine,
    and MoonshotPrimeEngine_UltraScale_Extreme into a cohesive framework for
    prime number analysis, visualization, and theoretical exploration.
    
    Key capabilities:
    1. Multi-scale prime detection (efficient for both small and ultra-ultra-large primes)
    2. Geometric analysis and visualization of prime patterns
    3. Adaptive mode selection based on input range
    4. Advanced mathematical model integration
    """
    
    # Mathematical constants
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    INV_PHI = 1 / PHI
    EULER = 0.57721566490153286060651209008240243104215933593992  # Euler's constant
    
    def __init__(self, mode='adaptive', 
                 hybrid_params=None, 
                 moonshot_params=None,
                 ultrascale_params=None,
                 extreme_params=None,
                 auto_calibrate=True):
        """
        Initialize the unified framework
        
        Parameters:
        - mode: Operation mode ('hybrid', 'moonshot', 'ultrascale', 'extreme', or 'adaptive')
        - hybrid_params: Parameters for HybridPrimeEngine
        - moonshot_params: Parameters for MoonshotPrimeEngine
        - ultrascale_params: Parameters for MoonshotPrimeEngine_UltraScale
        - extreme_params: Parameters for MoonshotPrimeEngine_UltraScale_Extreme
        - auto_calibrate: Whether to automatically calibrate models
        """
        self.mode = mode
        
        # Set default parameters if not provided
        if hybrid_params is None:
            hybrid_params = {
                'wave_range': 1000000,
                'fourier_k': 24,
                'alpha': 0.35,
                'window_width': 501,
                'valley_threshold': 0.963,
                'use_manifold': True,
                'use_bspline': True
            }
            
        if moonshot_params is None:
            moonshot_params = {
                'fractal_depth': 5,
                'resonance_dims': 5,
                'use_gpu': False
            }
            
        if ultrascale_params is None:
            ultrascale_params = {
                'fractal_depth': 4,
                'resonance_dims': 4,
                'use_parallel': True
            }
            
        if extreme_params is None:
            extreme_params = {
                'fractal_depth': 3,
                'resonance_dims': 3,
                'use_parallel': True
            }
            
        # Initialize engines based on mode
        if mode in ['hybrid', 'adaptive']:
            self._init_hybrid_engine(**hybrid_params)
            
        if mode in ['moonshot', 'adaptive']:
            self._init_moonshot_engine(**moonshot_params)
            
        if mode in ['ultrascale', 'adaptive']:
            self._init_ultrascale_engine(**ultrascale_params)
            
        if mode in ['extreme', 'adaptive']:
            self._init_extreme_engine(**extreme_params)
            
        # Initialize cubic manifold model
        self.manifold_model = None
        
        # Setup default thresholds for size-based decisions
        self.size_thresholds = {
            'small': 100000,            # Use standard tests
            'medium': 10000000,         # Use hybrid engine
            'large': 1000000000,        # Use moonshot engine
            'ultralarge': 1000000000000, # Use ultrascale engine 
            'extreme': 1000000000000000 # Use extreme engine
        }
        
        # Calibrate if requested
        if auto_calibrate:
            self.calibrate()
        
        # Initialize visualization settings
        self.viz_settings = {
            'color_map': 'viridis',
            'point_size': 20,
            'dpi': 150,
            'figsize': (10, 8)
        }
    
    def _init_hybrid_engine(self, **params):
        """Initialize the hybrid engine with geometric optimizations"""
        from HybridPrimeEngineV4_0 import HybridPrimeEngineV4_0
        
        print("Initializing HybridPrimeEngineV4.0...")
        try:
            self.hybrid_engine = HybridPrimeEngineV4_0(**params)
            print("Hybrid engine initialized successfully")
        except Exception as e:
            print(f"Error initializing hybrid engine: {e}")
            self.hybrid_engine = None
    
    def _init_moonshot_engine(self, **params):
        """Initialize the moonshot engine for large numbers"""
        from MoonshotPrimeEngine import MoonshotPrimeEngine
        
        print("Initializing MoonshotPrimeEngine...")
        try:
            self.moonshot_engine = MoonshotPrimeEngine(**params)
            print("Moonshot engine initialized successfully")
        except Exception as e:
            print(f"Error initializing moonshot engine: {e}")
            self.moonshot_engine = None
    
    def _init_ultrascale_engine(self, **params):
        """Initialize the UltraScale engine for ultra-large numbers"""
        from MoonshotPrimeEngine_UltraScale import MoonshotPrimeEngine_UltraScale
        
        print("Initializing MoonshotPrimeEngine_UltraScale...")
        try:
            self.ultrascale_engine = MoonshotPrimeEngine_UltraScale(**params)
            print("UltraScale engine initialized successfully")
        except Exception as e:
            print(f"Error initializing ultrascale engine: {e}")
            self.ultrascale_engine = None
    
    def _init_extreme_engine(self, **params):
        """Initialize the UltraScale Extreme engine for extreme-scale numbers"""
        from MoonshotPrimeEngine_UltraScale_Extreme import MoonshotPrimeEngine_UltraScale_Extreme
        
        print("Initializing MoonshotPrimeEngine_UltraScale_Extreme...")
        try:
            self.extreme_engine = MoonshotPrimeEngine_UltraScale_Extreme(**params)
            print("Extreme engine initialized successfully")
        except Exception as e:
            print(f"Error initializing extreme engine: {e}")
            self.extreme_engine = None
    
    def calibrate(self, sample_size=1000, max_range=1000000):
        """
        Calibrate all models for optimal performance
        
        Parameters:
        - sample_size: Number of samples for calibration
        - max_range: Maximum number range for calibration
        """
        print(f"Calibrating with {sample_size} samples up to {max_range}...")
        
        # Generate calibration data
        log_start, log_end = np.log10(2), np.log10(max_range)
        log_samples = np.random.uniform(log_start, log_end, sample_size)
        numbers = np.power(10, log_samples).astype(int)
        
        # Ensure uniqueness and sort
        numbers = sorted(list(set(numbers)))
        
        # Calibrate manifold model if not already calibrated
        if hasattr(self, 'hybrid_engine') and self.hybrid_engine is not None:
            if not self.hybrid_engine.use_manifold or self.hybrid_engine.manifold_coefficients is None:
                self.hybrid_engine._calibrate_manifold(sample_size=sample_size, max_range=max_range)
        
        # Build performance profile for adaptive mode
        if self.mode == 'adaptive':
            self._build_performance_profile(numbers)
    
    def _build_performance_profile(self, numbers):
        """
        Build performance profiles for adaptive mode decision-making
        
        Parameters:
        - numbers: List of sample numbers
        """
        # Skip if not in adaptive mode or missing engines
        if self.mode != 'adaptive':
            return
        
        # Select sample points across different ranges
        ranges = [
            (2, 1000),
            (1001, 10000),
            (10001, 100000),
            (100001, 1000000),
            (1000001, 10000000),
            (10000001, 100000000),
            (100000001, 1000000000)
        ]
        
        profiles = {}
        available_engines = []
        
        if hasattr(self, 'hybrid_engine') and self.hybrid_engine is not None:
            available_engines.append(('hybrid', self.hybrid_engine.is_prime))
            
        if hasattr(self, 'moonshot_engine') and self.moonshot_engine is not None:
            available_engines.append(('moonshot', self.moonshot_engine.is_prime_probabilistic))
            
        if hasattr(self, 'ultrascale_engine') and self.ultrascale_engine is not None:
            available_engines.append(('ultrascale', self.ultrascale_engine.is_prime_ultra))
            
        if hasattr(self, 'extreme_engine') and self.extreme_engine is not None:
            available_engines.append(('extreme', self.extreme_engine.is_prime_extreme))
        
        if not available_engines:
            print("No engines available for performance profiling.")
            return
        
        for range_start, range_end in ranges:
            # Filter numbers in this range
            range_numbers = [n for n in numbers if range_start <= n <= range_end]
            
            if len(range_numbers) < 5:
                continue
                
            # Sample to avoid too many tests
            if len(range_numbers) > 20:
                range_numbers = sorted(np.random.choice(range_numbers, 20, replace=False).tolist())
            
            profile = {name: {'time': 0, 'calls': 0} for name, _ in available_engines}
            
            # Test each engine
            for engine_name, engine_func in available_engines:
                start_time = time.time()
                for n in range_numbers:
                    _ = engine_func(n)
                profile[engine_name]['time'] = (time.time() - start_time) / len(range_numbers)
                profile[engine_name]['calls'] = len(range_numbers)
                
            # Store profile
            profiles[f"{range_start}-{range_end}"] = profile
        
        self.performance_profiles = profiles
        print("Performance profiles built for adaptive mode")
    
    def is_prime(self, n):
        """
        Determine if a number is prime using the appropriate engine
        
        Parameters:
        - n: Number to test
        
        Returns:
        - Boolean indicating primality
        """
        # Handle small primes directly
        if n < 2:
            return False
        if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
            return True
            
        # For larger numbers, select the appropriate engine
        if self.mode == 'hybrid' and self.hybrid_engine is not None:
            return self.hybrid_engine.is_prime(n)
            
        elif self.mode == 'moonshot' and self.moonshot_engine is not None:
            return self.moonshot_engine.is_prime_probabilistic(n)
            
        elif self.mode == 'ultrascale' and self.ultrascale_engine is not None:
            return self.ultrascale_engine.is_prime_ultra(n)
            
        elif self.mode == 'extreme' and self.extreme_engine is not None:
            return self.extreme_engine.is_prime_extreme(n)
            
        elif self.mode == 'adaptive':
            # Choose engine based on number size
            if n <= self.size_thresholds['small'] and self.hybrid_engine is not None:
                return self.hybrid_engine.is_prime(n)
            elif n <= self.size_thresholds['medium'] and self.hybrid_engine is not None:
                return self.hybrid_engine.is_prime(n)
            elif n <= self.size_thresholds['large'] and self.moonshot_engine is not None:
                return self.moonshot_engine.is_prime_probabilistic(n)
            elif n <= self.size_thresholds['ultralarge'] and self.ultrascale_engine is not None:
                return self.ultrascale_engine.is_prime_ultra(n)
            elif self.extreme_engine is not None:
                return self.extreme_engine.is_prime_extreme(n)
        
        # Fallback to basic primality test
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def prime_confidence(self, n):
        """
        Get confidence score for primality
        
        Parameters:
        - n: Number to evaluate
        
        Returns:
        - Confidence score (0-1)
        """
        if self.mode == 'hybrid' and self.hybrid_engine is not None:
            return self.hybrid_engine.prime_confidence(n)
        elif self.mode == 'moonshot' and self.moonshot_engine is not None:
            score = self.moonshot_engine.calculate_prime_score(n)
            return score
        elif self.mode == 'ultrascale' and self.ultrascale_engine is not None:
            return self.ultrascale_engine.calculate_prime_score_ultra(n)
        elif self.mode == 'extreme' and self.extreme_engine is not None:
            return self.extreme_engine.calculate_prime_score_extreme(n)
        elif self.mode == 'adaptive':
            if n <= self.size_thresholds['medium'] and self.hybrid_engine is not None:
                return self.hybrid_engine.prime_confidence(n)
            elif n <= self.size_thresholds['large'] and self.moonshot_engine is not None:
                return self.moonshot_engine.calculate_prime_score(n)
            elif n <= self.size_thresholds['ultralarge'] and self.ultrascale_engine is not None:
                return self.ultrascale_engine.calculate_prime_score_ultra(n)
            elif self.extreme_engine is not None:
                return self.extreme_engine.calculate_prime_score_extreme(n)
        
        # Fallback
        return 1.0 if self.is_prime(n) else 0.0
    
    def analyze_geometry(self, n_start, n_end, sample_size=1000):
        """
        Perform geometric analysis on a range of numbers
        
        Parameters:
        - n_start: Start of range
        - n_end: End of range
        - sample_size: Number of samples
        
        Returns:
        - Dictionary of analysis results
        """
        # Generate logarithmically spaced numbers
        log_start, log_end = np.log10(max(2, n_start)), np.log10(n_end)
        log_samples = np.random.uniform(log_start, log_end, sample_size)
        numbers = np.unique(np.power(10, log_samples).astype(int))
        
        # Ensure inclusion of important reference points
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        numbers = sorted(list(set(numbers).union(set(small_primes))))
        
        # Collect component scores
        primary_scores = []
        secondary_scores = []
        tertiary_scores = []
        is_prime_list = []
        
        print(f"Analyzing {len(numbers)} numbers...")
        
        for n in numbers:
            if self.hybrid_engine is not None:
                p_score, s_score, t_score = self.hybrid_engine.get_component_scores(n)
                is_prime_val = self.hybrid_engine.is_prime(n)
            else:
                # Fallback if hybrid engine isn't available
                # Use simpler scoring
                p_score = np.sin(2 * np.pi * np.sqrt(n))
                s_score = np.cos(2 * np.pi * np.sqrt(n))
                t_score = np.sin(2 * np.pi * np.log(n))
                is_prime_val = self.is_prime(n)
            
            primary_scores.append(p_score)
            secondary_scores.append(s_score)
            tertiary_scores.append(t_score)
            is_prime_list.append(is_prime_val)
        
        # Convert to arrays
        numbers = np.array(numbers)
        primary_scores = np.array(primary_scores)
        secondary_scores = np.array(secondary_scores)
        tertiary_scores = np.array(tertiary_scores)
        is_prime_list = np.array(is_prime_list)
        
        # Split into primes and composites
        primes = numbers[is_prime_list]
        composites = numbers[~is_prime_list]
        
        prime_p_scores = primary_scores[is_prime_list]
        prime_s_scores = secondary_scores[is_prime_list]
        prime_t_scores = tertiary_scores[is_prime_list]
        
        comp_p_scores = primary_scores[~is_prime_list]
        comp_s_scores = secondary_scores[~is_prime_list]
        comp_t_scores = tertiary_scores[~is_prime_list]
        
        # Fit cubic manifold
        log_n = np.log10(numbers)
        prime_log_n = np.log10(primes)
        
        # Linear model for prime sheet
        A = np.column_stack([prime_p_scores, prime_s_scores, np.ones_like(prime_p_scores)])
        b = prime_t_scores
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Extract coefficients
        coefficients = {
            'p_coeff': x[0],
            's_coeff': x[1],
            'intercept': x[2]
        }
        
        # Calculate ratio for golden ratio comparison
        p_s_ratio = abs(x[0] / x[1]) if x[1] != 0 else float('inf')
        
        # Results dictionary
        results = {
            'numbers': numbers.tolist(),
            'primes': primes.tolist(),
            'composites': composites.tolist(),
            'prime_count': len(primes),
            'composite_count': len(composites),
            'prime_sheet_coefficients': coefficients,
            'p_s_ratio': p_s_ratio,
            'phi_comparison': {
                'p_s_ratio': p_s_ratio,
                'inv_phi': self.INV_PHI,
                'difference': abs(p_s_ratio - self.INV_PHI),
                'percent_error': abs(p_s_ratio - self.INV_PHI) / self.INV_PHI * 100
            },
            'component_scores': {
                'primary': primary_scores.tolist(),
                'secondary': secondary_scores.tolist(),
                'tertiary': tertiary_scores.tolist()
            }
        }
        
        # Calculate additional metrics if we have hybrid engine
        if self.hybrid_engine is not None and hasattr(self.hybrid_engine, 'calculate_valley_score'):
            try:
                valley_scores = [self.hybrid_engine.calculate_valley_score(n) for n in numbers]
                avg_prime_valley = np.mean([valley_scores[i] for i, is_p in enumerate(is_prime_list) if is_p])
                avg_comp_valley = np.mean([valley_scores[i] for i, is_p in enumerate(is_prime_list) if not is_p])
                
                results['valley_scores'] = {
                    'avg_prime_valley': float(avg_prime_valley),
                    'avg_comp_valley': float(avg_comp_valley),
                    'ratio': float(avg_comp_valley / avg_prime_valley) if avg_prime_valley > 0 else float('inf')
                }
            except Exception as e:
                print(f"Error calculating valley scores: {e}")
        
        return results
    
    def visualize_prime_sheet(self, n_start, n_end, sample_size=1000, ax=None, save_path=None):
        """
        Visualize the prime sheet in 3D space
        
        Parameters:
        - n_start: Start of range
        - n_end: End of range
        - sample_size: Number of samples
        - ax: Matplotlib axis (optional)
        - save_path: Path to save figure (optional)
        
        Returns:
        - Matplotlib figure
        """
        results = self.analyze_geometry(n_start, n_end, sample_size)
        
        # Extract data
        numbers = np.array(results['numbers'])
        is_prime = np.array([n in results['primes'] for n in numbers])
        
        p_scores = np.array(results['component_scores']['primary'])
        s_scores = np.array(results['component_scores']['secondary'])
        t_scores = np.array(results['component_scores']['tertiary'])
        
        # Create the plot
        if ax is None:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=self.viz_settings['figsize'], dpi=self.viz_settings['dpi'])
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Plot points
        ax.scatter(
            p_scores[is_prime], s_scores[is_prime], t_scores[is_prime],
            c='blue', label='Prime', s=self.viz_settings['point_size'], alpha=0.8
        )
        ax.scatter(
            p_scores[~is_prime], s_scores[~is_prime], t_scores[~is_prime],
            c='red', label='Composite', s=self.viz_settings['point_size'], alpha=0.3
        )
        
        # Plot the prime sheet
        coef = results['prime_sheet_coefficients']
        x_range = np.linspace(min(p_scores), max(p_scores), 10)
        y_range = np.linspace(min(s_scores), max(s_scores), 10)
        X, Y = np.meshgrid(x_range, y_range)
        Z = coef['p_coeff'] * X + coef['s_coeff'] * Y + coef['intercept']
        
        ax.plot_surface(X, Y, Z, alpha=0.3, color='green')
        
        # Add labels and title
        ax.set_xlabel('Primary Score')
        ax.set_ylabel('Secondary Score')
        ax.set_zlabel('Tertiary Score')
        ax.set_title(f'Prime Sheet (Range: {n_start}-{n_end})')
        
        # Add equation and golden ratio info
        ratio_info = results['phi_comparison']
        equation_text = f"z = {coef['p_coeff']:.3f}x + {coef['s_coeff']:.3f}y + {coef['intercept']:.3f}"
        ratio_text = f"Coefficient ratio: {ratio_info['p_s_ratio']:.3f} (1/φ ≈ {self.INV_PHI:.3f})"
        error_text = f"Error: {ratio_info['percent_error']:.2f}%"
        
        ax.text2D(0.05, 0.95, equation_text, transform=ax.transAxes)
        ax.text2D(0.05, 0.90, ratio_text, transform=ax.transAxes)
        ax.text2D(0.05, 0.85, error_text, transform=ax.transAxes)
        
        # Add legend
        ax.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.viz_settings['dpi'], bbox_inches='tight')
        
        return fig
    
    def visualize_two_circles(self, n_start, n_end, sample_size=1000, ax=None, save_path=None):
        """
        Visualize the two-circles model with golden ratio relationship
        
        Parameters:
        - n_start: Start of range
        - n_end: End of range  
        - sample_size: Number of samples
        - ax: Matplotlib axis (optional)
        - save_path: Path to save figure (optional)
        
        Returns:
        - Matplotlib figure
        """
        results = self.analyze_geometry(n_start, n_end, sample_size)
        
        # Extract data
        numbers = np.array(results['numbers'])
        primes = np.array(results['primes'])
        composites = np.array(results['composites'])
        
        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(figsize=self.viz_settings['figsize'], dpi=self.viz_settings['dpi'])
        else:
            fig = ax.figure
        
        # Draw the two circles
        # Large circle (radius 1)
        theta = np.linspace(0, 2*np.pi, 100)
        x1 = np.cos(theta)
        y1 = np.sin(theta)
        ax.plot(x1, y1, 'b-', alpha=0.5, label='Prime Circle (r=1)')
        
        # Small circle (radius 1/φ)
        x2 = self.INV_PHI * np.cos(theta)
        y2 = self.INV_PHI * np.sin(theta) + (1 - self.INV_PHI)  # Positioned to be tangent
        ax.plot(x2, y2, 'r-', alpha=0.5, label='Composite Circle (r=1/φ)')
        
        # Plot points along the circles
        # We'll place each number at an angle based on its log value
        for n in primes:
            angle = (np.log(n) * self.PHI) % (2*np.pi)
            x = np.cos(angle)
            y = np.sin(angle)
            ax.scatter(x, y, c='blue', s=self.viz_settings['point_size']/2, alpha=0.8)
        
        for n in composites:
            angle = (np.log(n) * self.PHI) % (2*np.pi)
            x = self.INV_PHI * np.cos(angle)
            y = self.INV_PHI * np.sin(angle) + (1 - self.INV_PHI)
            ax.scatter(x, y, c='red', s=self.viz_settings['point_size']/2, alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Two-Circle Model (Range: {n_start}-{n_end})')
        
        # Mark the tangent point
        ax.scatter([0], [1], c='green', s=100, marker='x', label='Tangent Point')
        
        # Add special angle markers (23° slip gate)
        slip_gate_angle = np.radians(25)
        ax.plot([0, np.cos(slip_gate_angle)], [1, 1-np.sin(slip_gate_angle)], 'g--', alpha=0.7)
        ax.plot([0, np.cos(-slip_gate_angle)], [1, 1-np.sin(-slip_gate_angle)], 'g--', alpha=0.7)
        
        # Add golden ratio info
        ax.text(0.05, 0.95, f"Circle ratio: 1 : {self.INV_PHI:.4f}", transform=ax.transAxes)
        ax.text(0.05, 0.90, f"Golden ratio: φ = {self.PHI:.4f}", transform=ax.transAxes)
        ax.text(0.05, 0.85, f"Slip gate angle: ±25°", transform=ax.transAxes)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.7)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.viz_settings['dpi'], bbox_inches='tight')
        
        return fig
    
    def benchmark(self, ranges=None, sample_size=100):
        """
        Run benchmark tests on different number ranges
        
        Parameters:
        - ranges: List of (start, end) tuples for ranges to test
        - sample_size: Number of samples per range
        
        Returns:
        - Dictionary of benchmark results
        """
        if ranges is None:
            ranges = [
                (2, 1000),
                (1001, 10000),
                (10001, 100000),
                (100001, 1000000),
                (1000001, 10000000),
                (10000001, 100000000),
                (100000001, 1000000000),
                (1000000001, 10000000000)
            ]
        
        results = {}
        
        for start, end in ranges:
            print(f"Benchmarking range {start}-{end}...")
            
            # Generate logarithmically spaced numbers
            log_start, log_end = np.log10(max(2, start)), np.log10(end)
            log_samples = np.random.uniform(log_start, log_end, sample_size)
            numbers = np.unique(np.power(10, log_samples).astype(int))
            
            # Ensure we stay within the range
            numbers = [n for n in numbers if start <= n <= end]
            
            # Test different engines
            engines = []
            
            if hasattr(self, 'hybrid_engine') and self.hybrid_engine is not None:
                engines.append(('hybrid', self.hybrid_engine.is_prime))
                
            if hasattr(self, 'moonshot_engine') and self.moonshot_engine is not None:
                engines.append(('moonshot', self.moonshot_engine.is_prime_probabilistic))
                
            if hasattr(self, 'ultrascale_engine') and self.ultrascale_engine is not None:
                engines.append(('ultrascale', self.ultrascale_engine.is_prime_ultra))
                
            if hasattr(self, 'extreme_engine') and self.extreme_engine is not None:
                engines.append(('extreme', self.extreme_engine.is_prime_extreme))
                
            engines.append(('unified', self.is_prime))
            
            # Run tests
            range_results = {}
            
            for engine_name, prime_func in engines:
                # Measure time
                start_time = time.time()
                
                # Run tests
                results_list = []
                for n in numbers:
                    is_prime = prime_func(n)
                    results_list.append(is_prime)
                
                # Calculate time
                elapsed = time.time() - start_time
                avg_time = elapsed / len(numbers)
                
                # Store results
                range_results[engine_name] = {
                    'time': elapsed,
                    'avg_time': avg_time,
                    'count': len(numbers)
                }
                
            # Store range results
            results[f"{start}-{end}"] = range_results
        
        return results
    
    def benchmark_extreme_ranges(self, include_verification=False):
        """
        Specialized benchmark for extremely large numbers
        
        Parameters:
        - include_verification: Whether to verify with known primality test (warning: very slow)
        
        Returns:
        - Dictionary of benchmark results
        """
        # Check if we have the extreme engine
        if not hasattr(self, 'extreme_engine') or self.extreme_engine is None:
            print("Extreme engine not available for benchmarking.")
            return {}
        
        # Define extreme ranges to test
        extreme_ranges = [
            (10**9, 10**10),         # 1-10 billion
            (10**11, 10**12),        # 100B-1T
            (10**13, 10**14),        # 10T-100T
            (10**15, 10**16)         # 1-10 quadrillion
        ]
        
        # For each range, test only a few carefully selected numbers
        results = {}
        
        for start, end in extreme_ranges:
            print(f"Benchmarking extreme range {start}-{end}...")
            
            # Generate 3 logarithmically spaced numbers
            log_start, log_end = np.log10(start), np.log10(end)
            sample_points = np.linspace(log_start, log_end, 3)
            numbers = np.power(10, sample_points).astype(np.int64)
            
            # Test available engines
            available_engines = []
            
            if hasattr(self, 'ultrascale_engine') and self.ultrascale_engine is not None:
                available_engines.append(('ultrascale', self.ultrascale_engine.is_prime_ultra))
                
            available_engines.append(('extreme', self.extreme_engine.is_prime_extreme))
            available_engines.append(('unified', self.is_prime))
            
            # Run tests
            range_results = {}
            
            for engine_name, prime_func in available_engines:
                # Measure time
                start_time = time.time()
                
                # Run tests with individual timing
                detailed_results = []
                for n in numbers:
                    n_start_time = time.time()
                    is_prime = prime_func(n)
                    n_time = time.time() - n_start_time
                    
                    # Get verification if requested (this could be very slow)
                    verified = None
                    if include_verification and n < 10**12:
                        try:
                            import sympy
                            verified = sympy.isprime(n)
                        except Exception:
                            pass
                    
                    detailed_results.append({
                        'number': int(n),
                        'is_prime': is_prime,
                        'verified': verified,
                        'correct': is_prime == verified if verified is not None else None,
                        'time': n_time
                    })
                
                # Calculate time
                elapsed = time.time() - start_time
                avg_time = elapsed / len(numbers)
                
                # Store results
                range_results[engine_name] = {
                    'time': elapsed,
                    'avg_time': avg_time,
                    'count': len(numbers),
                    'detailed': detailed_results
                }
                
            # Print detailed results
            for engine_name, data in range_results.items():
                print(f"\n  Engine: {engine_name}")
                for detail in data['detailed']:
                    verified_str = ""
                    if detail['verified'] is not None:
                        verified_str = f", Verified: {detail['verified']}, Correct: {detail['correct']}"
                    print(f"    {detail['number']}: Prime={detail['is_prime']}, Time={detail['time']:.6f}s{verified_str}")
                    
            # Store range results
            results[f"{start}-{end}"] = range_results
        
        return results
    
    def visualize_scale_comparison(self, save_path='scale_comparison.png'):
        """
        Visualize scaling capabilities across different engines
        
        Parameters:
        - save_path: Path to save visualization
        """
        # Check available engines
        engines = []
        
        if hasattr(self, 'ultrascale_engine') and self.ultrascale_engine is not None:
            engines.append(('UltraScale', self.ultrascale_engine._adaptive_rescale))
            
        if hasattr(self, 'extreme_engine') and self.extreme_engine is not None:
            engines.append(('Extreme', self.extreme_engine._multi_layer_rescale))
            
        if not engines:
            print("No scaling engines available for visualization.")
            return
        
        # Generate logarithmically spaced numbers from 10^6 to 10^18
        exponents = np.linspace(6, 18, 100)
        numbers = np.power(10, exponents).astype(np.float64)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot each engine's scaling curve
        for name, scale_func in engines:
            scale_factors = [scale_func(float(n)) for n in numbers]
            plt.loglog(numbers, scale_factors, linewidth=3, label=f'{name} Engine')
        
        # Add threshold markers
        for name, threshold in self.size_thresholds.items():
            if 10**6 <= threshold <= 10**18:
                plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
                plt.text(threshold*1.1, 0.01, name, rotation=90, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Number Magnitude', fontsize=14)
        plt.ylabel('Scale Factor', fontsize=14)
        plt.title('Comparison of Scaling Capabilities Across Engines', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scale comparison visualization saved to {save_path}")

# Helper methods
def run_demo():
    """
    Run a demonstration of the framework's capabilities
    """
    # Initialize the framework with all engines
    framework = UnifiedPrimeGeometryFramework(mode='adaptive')
    
    # Test primality for different ranges
    test_numbers = [
        17, 91, 1009, 10007, 100003, 1000003,
        10000000019, 100000000003, 1000000000039
    ]
    
    print("Primality tests:")
    for n in test_numbers:
        is_prime = framework.is_prime(n)
        confidence = framework.prime_confidence(n)
        print(f"Number: {n}, Is Prime: {is_prime}, Confidence: {confidence:.4f}")
    
    # Visualize prime sheet for small numbers
    print("\nVisualizing prime sheet...")
    fig = framework.visualize_prime_sheet(10, 1000, sample_size=200)
    plt.savefig('prime_sheet_visualization.png')
    
    # Run quick benchmark
    print("\nRunning benchmark...")
    benchmark_results = framework.benchmark(
        ranges=[(2, 1000), (1001, 10000), (10**6, 10**7)],
        sample_size=20
    )
    
    # Print benchmark results
    for range_key, range_data in benchmark_results.items():
        print(f"\nRange: {range_key}")
        for engine, data in range_data.items():
            print(f"  {engine}: {data['avg_time']:.6f} sec/number")
    
    # Visualize scaling comparison if available
    if (hasattr(framework, 'ultrascale_engine') and framework.ultrascale_engine is not None or
        hasattr(framework, 'extreme_engine') and framework.extreme_engine is not None):
        framework.visualize_scale_comparison()
    
    # Run extreme benchmark if available
    if hasattr(framework, 'extreme_engine') and framework.extreme_engine is not None:
        print("\nRunning extreme range benchmark...")
        extreme_results = framework.benchmark_extreme_ranges()
    
    print("\nDemo complete!")

if __name__ == "__main__":
    run_demo() 