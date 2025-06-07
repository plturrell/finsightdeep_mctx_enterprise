#!/usr/bin/env python3
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Comprehensive CPU vs GPU Benchmark Suite for MCTX.

This script runs a comprehensive set of benchmarks comparing the performance
of MCTX on CPU vs GPU (with various optimizations), covering different
search scenarios, batch sizes, and workloads.

The benchmark results are saved to a JSON file for further analysis and 
visualization.
"""

import os
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import platform
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp

import mctx
from mctx._src import base
from mctx._src import tree as tree_lib
from mctx._src import t4_optimizations
from mctx._src import t4_search
from mctx.monitoring.metrics import MCTSMetricsCollector, SearchMetrics


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    batch_sizes: List[int]
    num_simulations: int
    num_actions: int
    tree_depth: int
    state_size: int
    use_gpu: bool
    use_t4_optimizations: bool
    use_mixed_precision: bool
    use_tensor_cores: bool
    repetitions: int = 5
    warmup_runs: int = 2


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: Dict[str, Any]
    execution_times: Dict[str, List[float]] = field(default_factory=dict)
    avg_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0
    std_execution_time: float = 0.0
    speedup_vs_cpu: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput: Optional[float] = None  # simulations per second
    metrics: Optional[Dict[str, Any]] = None


class BenchmarkSuite:
    """Comprehensive benchmark suite for MCTX on CPU and GPU."""
    
    def __init__(self, 
                 output_path: str = "benchmark_results",
                 collect_metrics: bool = True):
        """Initialize the benchmark suite.
        
        Args:
            output_path: Directory to save benchmark results
            collect_metrics: Whether to collect detailed performance metrics
        """
        self.output_path = output_path
        self.collect_metrics = collect_metrics
        self.results = []
        self.device_info = self._get_device_info()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize metrics collector if needed
        self.metrics_collector = MCTSMetricsCollector() if collect_metrics else None
        
        print(f"Benchmark suite initialized. Device info:")
        for key, value in self.device_info.items():
            print(f"  {key}: {value}")
    
    def _get_device_info(self) -> Dict[str, str]:
        """Get information about the available devices."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "cpu_info": platform.processor() or "Unknown",
            "num_cpu_cores": os.cpu_count() or 0,
            "has_gpu": len(jax.devices("gpu")) > 0,
            "num_gpus": len(jax.devices("gpu")),
        }
        
        # Try to get more detailed GPU info if available
        if info["has_gpu"]:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["gpu_name"] = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                info["gpu_memory"] = f"{pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3):.2f} GB"
                info["cuda_version"] = pynvml.nvmlSystemGetCudaDriverVersion() / 1000
                pynvml.nvmlShutdown()
            except (ImportError, Exception) as e:
                info["gpu_info_error"] = str(e)
                info["gpu_name"] = "Unknown"
        
        return info
    
    def run_all_benchmarks(self) -> None:
        """Run all benchmarks in the suite."""
        # Define standard benchmark configurations
        configs = []
        
        # Basic CPU benchmarks
        for batch_size in [1, 2, 4, 8, 16, 32]:
            configs.append(BenchmarkConfig(
                name=f"cpu_batch_{batch_size}",
                batch_sizes=[batch_size],
                num_simulations=100,
                num_actions=64,
                tree_depth=10,
                state_size=8,
                use_gpu=False,
                use_t4_optimizations=False,
                use_mixed_precision=False,
                use_tensor_cores=False
            ))
        
        # Basic GPU benchmarks (if available)
        if self.device_info["has_gpu"]:
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
                configs.append(BenchmarkConfig(
                    name=f"gpu_batch_{batch_size}",
                    batch_sizes=[batch_size],
                    num_simulations=100,
                    num_actions=64,
                    tree_depth=10,
                    state_size=8,
                    use_gpu=True,
                    use_t4_optimizations=False,
                    use_mixed_precision=False,
                    use_tensor_cores=False
                ))
            
            # T4-optimized benchmarks
            for batch_size in [16, 32, 64, 128, 256]:
                # FP32 with T4 optimizations
                configs.append(BenchmarkConfig(
                    name=f"t4_fp32_batch_{batch_size}",
                    batch_sizes=[batch_size],
                    num_simulations=100,
                    num_actions=64,
                    tree_depth=10,
                    state_size=8,
                    use_gpu=True,
                    use_t4_optimizations=True,
                    use_mixed_precision=False,
                    use_tensor_cores=False
                ))
                
                # FP16 with T4 optimizations
                configs.append(BenchmarkConfig(
                    name=f"t4_fp16_batch_{batch_size}",
                    batch_sizes=[batch_size],
                    num_simulations=100,
                    num_actions=64,
                    tree_depth=10,
                    state_size=8,
                    use_gpu=True,
                    use_t4_optimizations=True,
                    use_mixed_precision=True,
                    use_tensor_cores=False
                ))
                
                # Full T4 optimizations with tensor cores
                configs.append(BenchmarkConfig(
                    name=f"t4_full_batch_{batch_size}",
                    batch_sizes=[batch_size],
                    num_simulations=100,
                    num_actions=64,
                    tree_depth=10,
                    state_size=8,
                    use_gpu=True,
                    use_t4_optimizations=True,
                    use_mixed_precision=True,
                    use_tensor_cores=True
                ))
        
        # Run each benchmark configuration
        for config in configs:
            print(f"\nRunning benchmark: {config.name}")
            result = self.run_benchmark(config)
            self.results.append(result)
            
            # Print the result
            print(f"  Avg execution time: {result.avg_execution_time:.2f} ms")
            if result.speedup_vs_cpu is not None:
                print(f"  Speedup vs CPU: {result.speedup_vs_cpu:.2f}x")
            print(f"  Throughput: {result.throughput:.2f} simulations/second")
        
        # Save all results
        self.save_results()
        
        # Generate visualizations
        self.generate_visualizations()
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        # Create the benchmark model
        model = SimpleModel(config.num_actions, config.state_size)
        
        # Initialize result structure
        result = BenchmarkResult(config=asdict(config))
        result.execution_times = {str(bs): [] for bs in config.batch_sizes}
        
        # Set up the platform
        platform = "gpu" if config.use_gpu else "cpu"
        prev_platform = os.environ.get("JAX_PLATFORM_NAME", "")
        os.environ["JAX_PLATFORM_NAME"] = platform
        
        # For each batch size
        for batch_size in config.batch_sizes:
            # Warmup runs
            for _ in range(config.warmup_runs):
                self._run_single_search(
                    model=model, 
                    batch_size=batch_size,
                    num_simulations=config.num_simulations,
                    tree_depth=config.tree_depth,
                    use_t4=config.use_t4_optimizations,
                    precision="fp16" if config.use_mixed_precision else "fp32",
                    optimize_tensor_cores=config.use_tensor_cores
                )
            
            # Actual timed runs
            times = []
            for _ in range(config.repetitions):
                start_time = time.time()
                
                if self.collect_metrics:
                    self.metrics_collector.start_collection()
                
                search_result = self._run_single_search(
                    model=model, 
                    batch_size=batch_size,
                    num_simulations=config.num_simulations,
                    tree_depth=config.tree_depth,
                    use_t4=config.use_t4_optimizations,
                    precision="fp16" if config.use_mixed_precision else "fp32",
                    optimize_tensor_cores=config.use_tensor_cores
                )
                
                if self.collect_metrics:
                    # Update tree metrics
                    self.metrics_collector.update_tree_metrics(search_result)
                    metrics = self.metrics_collector.stop_collection()
                    if result.metrics is None:
                        result.metrics = metrics.as_dict()
                
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                times.append(execution_time_ms)
            
            # Record times for this batch size
            result.execution_times[str(batch_size)] = times
        
        # Calculate statistics
        all_times = [t for batch_times in result.execution_times.values() for t in batch_times]
        result.avg_execution_time = np.mean(all_times)
        result.min_execution_time = np.min(all_times)
        result.max_execution_time = np.max(all_times)
        result.std_execution_time = np.std(all_times)
        
        # Calculate throughput (simulations per second)
        total_simulations = sum(config.batch_sizes) * config.num_simulations * config.repetitions
        total_time_seconds = sum(all_times) / 1000
        result.throughput = total_simulations / total_time_seconds
        
        # Restore previous platform
        os.environ["JAX_PLATFORM_NAME"] = prev_platform
        
        return result
    
    def _run_single_search(self,
                          model: 'SimpleModel',
                          batch_size: int,
                          num_simulations: int,
                          tree_depth: int,
                          use_t4: bool = False,
                          precision: str = "fp32",
                          optimize_tensor_cores: bool = False) -> tree_lib.Tree:
        """Run a single search and return the result.
        
        Args:
            model: Model to use for the search
            batch_size: Batch size
            num_simulations: Number of simulations
            tree_depth: Maximum tree depth
            use_t4: Whether to use T4 optimizations
            precision: Precision to use (fp16 or fp32)
            optimize_tensor_cores: Whether to optimize for tensor cores
            
        Returns:
            Search result tree
        """
        # Initialize model parameters
        params = model.init(jax.random.PRNGKey(42))
        
        # Initialize dummy state
        dummy_state = jnp.zeros((batch_size, model.state_size))
        
        # Create root function
        def root_fn(params, rng_key, state):
            prior_logits, value = model.apply(params, state)
            return mctx.RootFnOutput(
                prior_logits=prior_logits,
                value=value,
                embedding=state)
        
        # Create recurrent function
        def recurrent_fn(params, rng_key, action, embedding):
            next_state = jnp.roll(embedding, 1, axis=-1)  # Simple state transition
            next_state = next_state.at[..., 0].set(action)
            prior_logits, value = model.apply(params, next_state)
            return (
                mctx.RecurrentFnOutput(
                    reward=jnp.ones_like(value) * 0.1,  # Small positive reward
                    discount=jnp.ones_like(value) * 0.99,  # Standard discount
                    prior_logits=prior_logits,
                    value=value),
                next_state)
        
        # Create key for search
        key = jax.random.PRNGKey(0)
        
        # Create root node
        root_key, apply_key = jax.random.split(key)
        root = root_fn(params, apply_key, dummy_state)
        
        # Run the search
        if use_t4:
            # Use T4-optimized search
            result = t4_search.t4_search(
                params=params,
                rng_key=root_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                max_depth=tree_depth,
                root_action_selection_fn=mctx.muzero_action_selection,
                interior_action_selection_fn=mctx.muzero_action_selection,
                precision=precision,
                tensor_core_aligned=True,
                monitor_memory=False,
                optimize_memory_layout=True,
                optimize_tensor_cores=optimize_tensor_cores
            )
        else:
            # Use standard search
            result = mctx.search(
                params=params,
                rng_key=root_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                max_depth=tree_depth,
                root_action_selection_fn=mctx.muzero_action_selection,
                interior_action_selection_fn=mctx.muzero_action_selection
            )
        
        return result
    
    def save_results(self) -> None:
        """Save benchmark results to JSON file."""
        # Calculate CPU reference for speedup calculations
        cpu_results = [r for r in self.results if not r.config["use_gpu"]]
        if cpu_results:
            cpu_reference = cpu_results[0].avg_execution_time
            
            # Update speedup values
            for result in self.results:
                if result.config["use_gpu"]:
                    result.speedup_vs_cpu = cpu_reference / result.avg_execution_time
        
        # Convert results to dictionary
        results_dict = {
            "device_info": self.device_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in self.results]
        }
        
        # Save to file
        output_file = os.path.join(self.output_path, "benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nBenchmark results saved to {output_file}")
    
    def generate_visualizations(self) -> None:
        """Generate visualizations of benchmark results."""
        # Create a dataframe for easier plotting
        rows = []
        for result in self.results:
            for batch_size, times in result.execution_times.items():
                for time_ms in times:
                    row = {
                        "name": result.config["name"],
                        "batch_size": int(batch_size),
                        "time_ms": time_ms,
                        "use_gpu": result.config["use_gpu"],
                        "use_t4": result.config["use_t4_optimizations"],
                        "mixed_precision": result.config["use_mixed_precision"],
                        "tensor_cores": result.config["use_tensor_cores"],
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Plot 1: Execution time by batch size and configuration
        plt.figure(figsize=(12, 8))
        
        # Group by configuration type
        cpu_df = df[~df["use_gpu"]]
        gpu_basic_df = df[df["use_gpu"] & ~df["use_t4"]]
        t4_fp32_df = df[df["use_gpu"] & df["use_t4"] & ~df["mixed_precision"]]
        t4_fp16_df = df[df["use_gpu"] & df["use_t4"] & df["mixed_precision"] & ~df["tensor_cores"]]
        t4_full_df = df[df["use_gpu"] & df["use_t4"] & df["mixed_precision"] & df["tensor_cores"]]
        
        # Calculate means by batch size for each configuration
        if not cpu_df.empty:
            cpu_means = cpu_df.groupby("batch_size")["time_ms"].mean()
            plt.plot(cpu_means.index, cpu_means.values, "o-", label="CPU")
        
        if not gpu_basic_df.empty:
            gpu_means = gpu_basic_df.groupby("batch_size")["time_ms"].mean()
            plt.plot(gpu_means.index, gpu_means.values, "s-", label="GPU (Basic)")
        
        if not t4_fp32_df.empty:
            t4_fp32_means = t4_fp32_df.groupby("batch_size")["time_ms"].mean()
            plt.plot(t4_fp32_means.index, t4_fp32_means.values, "^-", label="T4 (FP32)")
        
        if not t4_fp16_df.empty:
            t4_fp16_means = t4_fp16_df.groupby("batch_size")["time_ms"].mean()
            plt.plot(t4_fp16_means.index, t4_fp16_means.values, "d-", label="T4 (FP16)")
        
        if not t4_full_df.empty:
            t4_full_means = t4_full_df.groupby("batch_size")["time_ms"].mean()
            plt.plot(t4_full_means.index, t4_full_means.values, "*-", label="T4 (Full Optimizations)")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Execution Time (ms)")
        plt.title("MCTX Search Performance by Configuration")
        plt.grid(True)
        plt.legend()
        plt.xscale("log", base=2)
        plt.yscale("log")
        
        # Save the plot
        plt.savefig(os.path.join(self.output_path, "execution_time_comparison.png"))
        
        # Plot 2: Speedup vs CPU
        plt.figure(figsize=(12, 8))
        speedup_rows = []
        
        for result in self.results:
            if result.speedup_vs_cpu is not None:
                speedup_rows.append({
                    "name": result.config["name"],
                    "speedup": result.speedup_vs_cpu,
                    "use_gpu": result.config["use_gpu"],
                    "use_t4": result.config["use_t4_optimizations"],
                    "mixed_precision": result.config["use_mixed_precision"],
                    "tensor_cores": result.config["use_tensor_cores"],
                    "batch_size": result.config["batch_sizes"][0]
                })
        
        if speedup_rows:
            speedup_df = pd.DataFrame(speedup_rows)
            
            # Group by configuration type
            gpu_basic_speedup = speedup_df[speedup_df["use_gpu"] & ~speedup_df["use_t4"]]
            t4_fp32_speedup = speedup_df[speedup_df["use_gpu"] & speedup_df["use_t4"] & ~speedup_df["mixed_precision"]]
            t4_fp16_speedup = speedup_df[speedup_df["use_gpu"] & speedup_df["use_t4"] & speedup_df["mixed_precision"] & ~speedup_df["tensor_cores"]]
            t4_full_speedup = speedup_df[speedup_df["use_gpu"] & speedup_df["use_t4"] & speedup_df["mixed_precision"] & speedup_df["tensor_cores"]]
            
            # Plot speedups
            if not gpu_basic_speedup.empty:
                plt.plot(gpu_basic_speedup["batch_size"], gpu_basic_speedup["speedup"], "s-", label="GPU (Basic)")
            
            if not t4_fp32_speedup.empty:
                plt.plot(t4_fp32_speedup["batch_size"], t4_fp32_speedup["speedup"], "^-", label="T4 (FP32)")
            
            if not t4_fp16_speedup.empty:
                plt.plot(t4_fp16_speedup["batch_size"], t4_fp16_speedup["speedup"], "d-", label="T4 (FP16)")
            
            if not t4_full_speedup.empty:
                plt.plot(t4_full_speedup["batch_size"], t4_full_speedup["speedup"], "*-", label="T4 (Full Optimizations)")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Speedup vs. CPU")
            plt.title("MCTX Search Speedup Relative to CPU")
            plt.grid(True)
            plt.legend()
            plt.xscale("log", base=2)
            
            # Save the plot
            plt.savefig(os.path.join(self.output_path, "speedup_comparison.png"))
        
        print(f"Visualizations saved to {self.output_path}")


class SimpleModel:
    """Simple neural network model for benchmarking."""
    
    def __init__(self, num_actions: int, state_size: int = 8):
        """Initialize the model.
        
        Args:
            num_actions: Number of actions in the action space
            state_size: Size of the state representation
        """
        self.num_actions = num_actions
        self.state_size = state_size
    
    def init(self, rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize model parameters.
        
        Args:
            rng_key: Random key for initialization
            
        Returns:
            Tuple of (policy_params, value_params)
        """
        key1, key2 = jax.random.split(rng_key)
        policy_params = jax.random.normal(key1, (self.state_size, self.num_actions))
        value_params = jax.random.normal(key2, (self.state_size, 1))
        return (policy_params, value_params)
    
    def apply(self, params: Tuple[jnp.ndarray, jnp.ndarray], state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the model to a state.
        
        Args:
            params: Model parameters (policy_params, value_params)
            state: State representation
            
        Returns:
            Tuple of (policy_logits, value)
        """
        policy_params, value_params = params
        
        # Simple linear model
        policy_logits = jnp.dot(state, policy_params)
        value = jnp.dot(state, value_params).squeeze(-1)
        
        return policy_logits, value


def main():
    """Main function to run the benchmark suite."""
    parser = argparse.ArgumentParser(description="MCTX CPU vs GPU Benchmark Suite")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--collect-metrics", action="store_true",
                      help="Collect detailed performance metrics")
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate a detailed HTML report")
    args = parser.parse_args()
    
    # Check if GPU is available
    has_gpu = len(jax.devices("gpu")) > 0
    if not has_gpu:
        print("WARNING: No GPU detected. Only CPU benchmarks will be run.")
    
    # Run benchmarks
    suite = BenchmarkSuite(
        output_path=args.output_dir,
        collect_metrics=args.collect_metrics
    )
    suite.run_all_benchmarks()
    
    # Generate HTML report if requested
    if args.generate_report:
        try:
            generate_html_report(
                os.path.join(args.output_dir, "benchmark_results.json"),
                os.path.join(args.output_dir, "benchmark_report.html")
            )
            print(f"HTML report saved to {os.path.join(args.output_dir, 'benchmark_report.html')}")
        except Exception as e:
            print(f"Error generating HTML report: {e}")


def generate_html_report(input_json: str, output_html: str) -> None:
    """Generate an HTML report from benchmark results.
    
    Args:
        input_json: Path to input JSON file with benchmark results
        output_html: Path to output HTML file
    """
    # Load benchmark results
    with open(input_json, "r") as f:
        data = json.load(f)
    
    # Basic HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCTX CPU vs GPU Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ width: 100%; max-width: 800px; margin: 20px 0; }}
            .info {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; }}
        </style>
    </head>
    <body>
        <h1>MCTX CPU vs GPU Benchmark Report</h1>
        <p>Generated on: {data['timestamp']}</p>
        
        <h2>System Information</h2>
        <div class="info">
    """
    
    # Add device info
    html += "<table>"
    for key, value in data["device_info"].items():
        html += f"<tr><th>{key}</th><td>{value}</td></tr>"
    html += "</table></div>"
    
    # Add summary section
    html += "<h2>Summary</h2>"
    
    # Calculate summary statistics
    cpu_results = [r for r in data["results"] if not r["config"]["use_gpu"]]
    gpu_results = [r for r in data["results"] if r["config"]["use_gpu"] and not r["config"]["use_t4_optimizations"]]
    t4_fp32_results = [r for r in data["results"] if r["config"]["use_gpu"] and r["config"]["use_t4_optimizations"] and not r["config"]["use_mixed_precision"]]
    t4_fp16_results = [r for r in data["results"] if r["config"]["use_gpu"] and r["config"]["use_t4_optimizations"] and r["config"]["use_mixed_precision"] and not r["config"]["use_tensor_cores"]]
    t4_full_results = [r for r in data["results"] if r["config"]["use_gpu"] and r["config"]["use_t4_optimizations"] and r["config"]["use_mixed_precision"] and r["config"]["use_tensor_cores"]]
    
    # Get max speedups
    max_speedups = {
        "GPU (Basic)": max([r["speedup_vs_cpu"] for r in gpu_results], default=0),
        "T4 (FP32)": max([r["speedup_vs_cpu"] for r in t4_fp32_results], default=0),
        "T4 (FP16)": max([r["speedup_vs_cpu"] for r in t4_fp16_results], default=0),
        "T4 (Full Optimizations)": max([r["speedup_vs_cpu"] for r in t4_full_results], default=0)
    }
    
    html += "<table>"
    html += "<tr><th>Configuration</th><th>Max Speedup vs CPU</th></tr>"
    for config, speedup in max_speedups.items():
        html += f"<tr><td>{config}</td><td>{speedup:.2f}x</td></tr>"
    html += "</table>"
    
    # Add charts
    html += """
        <h2>Performance Charts</h2>
        <p>These charts are static images. For interactive visualization, use the benchmark_results.json file.</p>
        
        <h3>Execution Time Comparison</h3>
        <img class="chart" src="execution_time_comparison.png" alt="Execution Time Comparison">
        
        <h3>Speedup Comparison</h3>
        <img class="chart" src="speedup_comparison.png" alt="Speedup Comparison">
    """
    
    # Add detailed results
    html += "<h2>Detailed Results</h2>"
    html += "<table>"
    html += "<tr><th>Configuration</th><th>Batch Size</th><th>Avg Time (ms)</th><th>Speedup vs CPU</th><th>Throughput (sim/s)</th></tr>"
    
    for result in data["results"]:
        config_name = result["config"]["name"]
        batch_size = result["config"]["batch_sizes"][0]
        avg_time = result["avg_execution_time"]
        speedup = result["speedup_vs_cpu"] if result["speedup_vs_cpu"] is not None else "N/A"
        throughput = result["throughput"]
        
        html += f"<tr><td>{config_name}</td><td>{batch_size}</td><td>{avg_time:.2f}</td><td>{speedup if speedup == 'N/A' else f'{speedup:.2f}x'}</td><td>{throughput:.2f}</td></tr>"
    
    html += "</table>"
    
    # Close HTML document
    html += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_html, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()