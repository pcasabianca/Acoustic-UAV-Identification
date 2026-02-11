#!/usr/bin/env python3
"""
Difficulty Distribution Visualization

Analyzes and visualizes difficulty scores across the dataset.
Plugin for the viz_all.py orchestrator.
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add tools and repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tools.viz_core import VizContext, VizResult
import config

VIZ_NAME = "difficulty_distribution"
VIZ_ORDER = 100  # Run after other viz


def load_metadata_with_difficulty():
    """Load metadata files containing difficulty scores."""
    datasets = {
        'train': Path(config.DATASET_TRAIN_DIR) / 'metadata.jsonl',
        'val': Path(config.DATASET_VAL_DIR) / 'metadata.jsonl',
        'test': Path(config.DATASET_TEST_DIR) / 'metadata.jsonl'
    }
    
    all_samples = []
    for split_name, meta_path in datasets.items():
        if not meta_path.exists():
            continue
        
        with open(meta_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                sample['split'] = split_name
                all_samples.append(sample)
    
    return all_samples


def extract_difficulty_data(samples):
    """Extract difficulty scores and related metadata."""
    data = {
        'drone': {'difficulty': [], 'snr': [], 'distance': [], 'scenario': []},
        'ambient': {'difficulty': [], 'scenario': [], 'tonality': [], 'low_freq': []}
    }
    
    for sample in samples:
        if 'difficulty' not in sample:
            continue
        
        diff = sample['difficulty']
        label = sample.get('label', 0)
        scenario = sample.get('scenario', 'unknown')
        
        if label == 1:  # Drone
            data['drone']['difficulty'].append(diff)
            data['drone']['snr'].append(sample.get('snr_est_db', np.nan))
            data['drone']['distance'].append(sample.get('distance_m', np.nan))
            data['drone']['scenario'].append(scenario)
        else:  # Ambient
            data['ambient']['difficulty'].append(diff)
            data['ambient']['scenario'].append(scenario)
            # Extract ambient-specific metrics if available
            data['ambient']['tonality'].append(sample.get('tonality_score', np.nan))
            data['ambient']['low_freq'].append(sample.get('low_freq_ratio', np.nan))
    
    return data


def plot_difficulty_histograms(data, output_file):
    """Plot difficulty distribution histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Drone samples
    if data['drone']['difficulty']:
        axes[0].hist(data['drone']['difficulty'], bins=30, alpha=0.7, 
                     color='#e74c3c', edgecolor='black', density=True)
        axes[0].axvline(np.mean(data['drone']['difficulty']), 
                        color='darkred', linestyle='--', linewidth=2, 
                        label=f'Mean: {np.mean(data["drone"]["difficulty"]):.3f}')
        axes[0].set_xlabel('Difficulty Score', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Drone Samples Difficulty Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    
    # Ambient samples
    if data['ambient']['difficulty']:
        axes[1].hist(data['ambient']['difficulty'], bins=30, alpha=0.7, 
                     color='#3498db', edgecolor='black', density=True)
        axes[1].axvline(np.mean(data['ambient']['difficulty']), 
                        color='darkblue', linestyle='--', linewidth=2, 
                        label=f'Mean: {np.mean(data["ambient"]["difficulty"]):.3f}')
        axes[1].set_xlabel('Difficulty Score', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Ambient Samples Difficulty Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_snr_vs_difficulty(data, output_file):
    """Scatter plot of SNR vs difficulty for drone samples."""
    if not data['drone']['difficulty'] or not data['drone']['snr']:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    snr = np.array(data['drone']['snr'])
    diff = np.array(data['drone']['difficulty'])
    
    # Remove NaN values
    mask = ~(np.isnan(snr) | np.isnan(diff))
    snr = snr[mask]
    diff = diff[mask]
    
    if len(snr) == 0:
        plt.close()
        return None
    
    # Scatter with colormap
    scatter = ax.scatter(snr, diff, c=diff, cmap='RdYlGn_r', 
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    # Trend line
    z = np.polyfit(snr, diff, 1)
    p = np.poly1d(z)
    ax.plot(sorted(snr), p(sorted(snr)), "r--", alpha=0.8, linewidth=2, 
            label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Correlation
    corr = np.corrcoef(snr, diff)[0, 1]
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Difficulty Score', fontsize=12)
    ax.set_title(f'SNR vs Difficulty (Drone Samples)\nCorrelation: {corr:.3f}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Difficulty Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def plot_scenario_difficulty_boxplot(data, output_file):
    """Boxplot of difficulty by scenario, SORTED by mean difficulty."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Drone scenarios
    if data['drone']['difficulty']:
        scenario_diff = defaultdict(list)
        for diff, scenario in zip(data['drone']['difficulty'], data['drone']['scenario']):
            scenario_diff[scenario].append(diff)
        
        # Sort scenarios by mean difficulty (ascending)
        scenario_means = [(s, np.mean(diffs)) for s, diffs in scenario_diff.items()]
        scenario_means.sort(key=lambda x: x[1])
        scenarios = [s for s, _ in scenario_means]
        diff_lists = [scenario_diff[s] for s in scenarios]
        
        bp1 = axes[0].boxplot(diff_lists, tick_labels=scenarios, patch_artist=True, 
                               showmeans=True, meanline=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.6)
        
        axes[0].set_xlabel('Scenario (ordered by mean difficulty)', fontsize=12)
        axes[0].set_ylabel('Difficulty Score', fontsize=12)
        axes[0].set_title('Difficulty by Scenario (Drone)', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
    
    # Ambient scenarios
    if data['ambient']['difficulty']:
        scenario_diff = defaultdict(list)
        for diff, scenario in zip(data['ambient']['difficulty'], data['ambient']['scenario']):
            scenario_diff[scenario].append(diff)
        
        # Sort scenarios by mean difficulty (ascending)
        scenario_means = [(s, np.mean(diffs)) for s, diffs in scenario_diff.items()]
        scenario_means.sort(key=lambda x: x[1])
        scenarios = [s for s, _ in scenario_means]
        diff_lists = [scenario_diff[s] for s in scenarios]
        
        bp2 = axes[1].boxplot(diff_lists, tick_labels=scenarios, patch_artist=True, 
                               showmeans=True, meanline=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.6)
        
        axes[1].set_xlabel('Scenario (ordered by mean difficulty)', fontsize=12)
        axes[1].set_ylabel('Difficulty Score', fontsize=12)
        axes[1].set_title('Difficulty by Scenario (Ambient)', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ambient_metric_vs_difficulty(data, output_file):
    """Scatter plot of low-frequency ratio vs difficulty for ambient samples."""
    if not data['ambient']['difficulty']:
        return None
    
    # Use low_freq ratio if available, otherwise tonality
    metric_name = 'Low-Freq Ratio (100-500 Hz)'
    metric_data = data['ambient']['low_freq']
    
    # Fallback to tonality if low_freq not available
    if all(np.isnan(metric_data)):
        metric_name = 'Tonality Score'
        metric_data = data['ambient']['tonality']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    metric = np.array(metric_data)
    diff = np.array(data['ambient']['difficulty'])
    
    # Remove NaN values
    mask = ~(np.isnan(metric) | np.isnan(diff))
    metric = metric[mask]
    diff = diff[mask]
    
    if len(metric) == 0:
        plt.close()
        return None
    
    # Scatter with colormap
    scatter = ax.scatter(metric, diff, c=diff, cmap='YlOrRd', 
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    # Trend line
    if len(metric) > 1:
        z = np.polyfit(metric, diff, 1)
        p = np.poly1d(z)
        ax.plot(sorted(metric), p(sorted(metric)), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Correlation
        corr = np.corrcoef(metric, diff)[0, 1]
        title_text = f'{metric_name} vs Difficulty (Ambient Samples)\nCorrelation: {corr:.3f}'
    else:
        title_text = f'{metric_name} vs Difficulty (Ambient Samples)'
    
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Difficulty Score', fontsize=12)
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    if len(metric) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Difficulty Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def print_difficulty_statistics(data, logger):
    """Print statistical summary of difficulty scores."""
    logger.info("="*80)
    logger.info("  DIFFICULTY STATISTICS")
    logger.info("="*80)
    
    for class_name, class_data in data.items():
        if not class_data['difficulty']:
            continue
        
        diff = np.array(class_data['difficulty'])
        logger.info(f"\n{class_name.upper()} SAMPLES:")
        logger.info(f"  Count:      {len(diff)}")
        logger.info(f"  Mean:       {np.mean(diff):.4f}")
        logger.info(f"  Std Dev:    {np.std(diff):.4f}")
        logger.info(f"  Min:        {np.min(diff):.4f}")
        logger.info(f"  25th %ile:  {np.percentile(diff, 25):.4f}")
        logger.info(f"  Median:     {np.median(diff):.4f}")
        logger.info(f"  75th %ile:  {np.percentile(diff, 75):.4f}")
        logger.info(f"  Max:        {np.max(diff):.4f}")
        
        # Binning
        bins = [0.0, 0.33, 0.67, 1.0]
        labels = ['Easy', 'Medium', 'Hard']
        counts = [np.sum((diff >= bins[i]) & (diff < bins[i+1])) for i in range(len(bins)-1)]
        counts[-1] += np.sum(diff >= bins[-1])  # Include max in last bin
        
        logger.info(f"\n  Difficulty Bins:")
        for label, count in zip(labels, counts):
            pct = 100 * count / len(diff) if len(diff) > 0 else 0
            logger.info(f"    {label:8s}: {count:5d} ({pct:5.1f}%)")


def run(ctx, args=None):
    """
    Generate difficulty distribution visualizations.
    
    Args:
        ctx: Visualization context with paths and configuration
        args: Optional additional arguments (unused)
        
    Returns:
        VizResult with generated artifacts
    """
    result = VizResult()
    
    try:
        # Load metadata
        ctx.logger.info("Loading metadata with difficulty scores...")
        samples = load_metadata_with_difficulty()
        
        if not samples:
            ctx.logger.warning("No metadata found! Skipping difficulty visualization.")
            result.skipped = True
            return result
        
        ctx.logger.info(f"Loaded {len(samples)} samples")
        
        # Extract difficulty data
        ctx.logger.info("Extracting difficulty data...")
        data = extract_difficulty_data(samples)
        
        drone_count = len(data['drone']['difficulty'])
        ambient_count = len(data['ambient']['difficulty'])
        
        if drone_count == 0 and ambient_count == 0:
            ctx.logger.warning("No difficulty scores found in metadata!")
            ctx.logger.info("Ensure difficulty scoring is enabled in build_config.json")
            result.skipped = True
            return result
        
        ctx.logger.info(f"Found {drone_count} drone samples, {ambient_count} ambient samples with difficulty scores")
        
        # Print statistics
        print_difficulty_statistics(data, ctx.logger)
        
        # Create output directory
        output_dir = ctx.out_dir / "difficulty_dataset"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate visualizations
        ctx.logger.info("Generating visualizations...")
        
        histograms = output_dir / "difficulty_histograms.png"
        plot_difficulty_histograms(data, histograms)
        result.artifacts.append(histograms)
        ctx.logger.info(f"Generated: {histograms}")
        
        snr_scatter = output_dir / "snr_vs_difficulty.png"
        if plot_snr_vs_difficulty(data, snr_scatter):
            result.artifacts.append(snr_scatter)
            ctx.logger.info(f"Generated: {snr_scatter}")
        
        boxplots = output_dir / "difficulty_by_scenario.png"
        plot_scenario_difficulty_boxplot(data, boxplots)
        result.artifacts.append(boxplots)
        ctx.logger.info(f"Generated: {boxplots}")
        
        # Ambient metric scatter plot
        ambient_scatter = output_dir / "ambient_metric_vs_difficulty.png"
        if plot_ambient_metric_vs_difficulty(data, ambient_scatter):
            result.artifacts.append(ambient_scatter)
            ctx.logger.info(f"Generated: {ambient_scatter}")
        
        # Summary stats
        result.summary = {
            'drone_count': drone_count,
            'ambient_count': ambient_count,
            'drone_mean_difficulty': float(np.mean(data['drone']['difficulty'])) if drone_count > 0 else None,
            'ambient_mean_difficulty': float(np.mean(data['ambient']['difficulty'])) if ambient_count > 0 else None
        }
        
    except Exception as e:
        ctx.logger.error(f"Error in difficulty visualization: {e}", exc_info=True)
        result.errors.append(str(e))
    
    return result


if __name__ == '__main__':
    # Standalone mode for testing
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    ctx = VizContext(
        run_id='standalone',
        repo_root=Path(__file__).parent.parent.parent,
        perf_dir=Path(config.PERFORMANCE_DIR),
        results_dir=Path(config.RESULTS_DIR),
        out_dir=Path(config.VIZ_OUTPUT_DIR) / 'difficulty_test',
        thresholds_path=None,
        perfs=[],
        models=[],
        splits=['train', 'val', 'test'],
        logger=logging.getLogger('difficulty_viz')
    )
    
    result = run(ctx)
    print(f"\nGenerated {len(result.artifacts)} artifacts")
    print(f"Summary: {result.summary}")
