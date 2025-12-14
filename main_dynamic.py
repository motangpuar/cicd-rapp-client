import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time
import numpy as np  # Added for safer math

# Configuration
RAPP_URL = "http://192.168.8.35:5000"
NFO_URL = "http://192.168.8.35:8080"
INSTANCE_ID = "be0cc18b-1b8e-4b58-a2bc-9f4681ac6142"
IPERF_BITRATE = 200  # Mbps - target bitrate for all tests
IPERF_DURATION = 10
PERF_DURATION = 15

# Test matrix - As provided in your latest file
TEST_SCENARIOS = [
    {
        "test_id": "P29",
        "oru": "pegatron",
        "branch": "starlingx/pegatron",
        "T1a_cp_dl": {"min": 285, "max": 429},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 180},
        "runs": 1
    },
    {
        "test_id": "P30",
        "oru": "pegatron",
        "branch": "starlingx/pegatron",
        "T1a_cp_dl": {"min": 285, "max": 470},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 180},
        "runs": 1
    },
    {
        "test_id": "P31",
        "oru": "pegatron",
        "branch": "starlingx/pegatron",
        "T1a_cp_dl": {"min": 285, "max": 550},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 280},
        "runs": 1
    },
    {
        "test_id": "P32",
        "oru": "liteon",
        "branch": "starlingx/liteon",
        "T1a_cp_dl": {"min": 285, "max": 429},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 180},
        "runs": 1
    },
    {
        "test_id": "P33",
        "oru": "liteon",
        "branch": "starlingx/liteon",
        "T1a_cp_dl": {"min": 285, "max": 470},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 180},
        "runs": 1
    },
    {
        "test_id": "P34",
        "oru": "liteon",
        "branch": "starlingx/liteon",
        "T1a_cp_dl": {"min": 285, "max": 550},
        "T1a_cp_ul": {"min": 285, "max": 429},
        "T1a_up": {"min": 125, "max": 350},
        "Ta4": {"min": 110, "max": 280},
        "runs": 1
    }
]

def build_gnb_config(scenario):
    """Build gNB deployment config for given scenario"""
    return {
        "name": f"oai-gnb-{scenario['test_id'].lower()}",
        "description": f"Test {scenario['test_id']}: {scenario['oru'].upper()} O-RU",
        "profile_type": "kubernetes",
        "artifact_repo_url": "https://github.com/motangpuar/ocloud-helm-templates.git",
        "artifact_name": "oai-gnb-fhi-72",
        "artifact_repo_branch": scenario['branch'],
        "target_cluster": "cc1397ba-b1c4-4a3e-bc8d-6af58ef53818",
        "values": {
            "fhi72": {
                "dpdk_iova_mode": "VA",
                "file_prefix": "fhi_72",
                "T1a_cp_dl": scenario['T1a_cp_dl'],
                "T1a_cp_ul": scenario['T1a_cp_ul'],
                "T1a_up": scenario['T1a_up'],
                "Ta4": scenario['Ta4'],
                "ru_config": {
                    "iq_width": 9,
                    "iq_width_prach": 9
                },
                "prach_config": {
                    "eAxC_offset": 4,
                    "kbar": 4
                }
            }
        }
    }

def deploy_gnb(config):
    """Deploy gNB via rApp (which forwards to NFO)"""
    print(f"  Deploying gNB...")
    try:
        resp = requests.post(f"{RAPP_URL}/nfo/deploy", json=config, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            deployment_id = data.get('instance_id')
            print(f"    ✓ Deployed: {deployment_id}")
            return deployment_id
        else:
            print(f"    ✗ Deploy failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"    ✗ Deploy error: {e}")
        return None

def terminate_gnb(deployment_id):
    """Terminate gNB via rApp"""
    print(f"  Terminating gNB {deployment_id}...")
    try:
        resp = requests.post(f"{NFO_URL}/api/o2dms/v2/deployments/{deployment_id}/terminate/", timeout=60)
        if resp.status_code == 200:
            print("    ✓ Terminated")
            return True
        else:
            print(f"    ✗ Terminate failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"    ✗ Terminate error: {e}")
        return False

def check_ue_status():
    """Check if UE is attached and get signal metrics"""
    try:
        resp = requests.get(f"{RAPP_URL}/ue/status", timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            print("STATUS UE:")
            print(status)
            attached = status.get('attached', False)
            has_data_ip = status.get('data_ip') is not None
            return attached and has_data_ip, status
        else:
            return False, None
    except Exception as e:
        print(f"    ✗ UE status error: {e}")
        return False, None

def get_ptp_status():
    """Get PTP synchronization metrics"""
    try:
        resp = requests.get(f"{RAPP_URL}/sideload/{INSTANCE_ID}/ptp_status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            offset = data.get('ptp_offset', '0')
            try:
                ptp_rms = abs(float(offset))
            except:
                ptp_rms = None
            return ptp_rms
        else:
            return None
    except Exception as e:
        print(f"    ✗ PTP error: {e}")
        return None

def check_fh_connection():
    """Check fronthaul connection status"""
    try:
        resp = requests.get(f"{RAPP_URL}/gnb/logs", timeout=10)
        if resp.status_code == 200:
            logs = resp.text
            connected = "FH connection established" in logs or "O-RU connected" in logs
            return connected
        else:
            return None
    except Exception as e:
        return None

def toggle_airplane_mode():
    """Toggle airplane mode off->on->off"""
    try:
        requests.post(f"{RAPP_URL}/ue/airplane/on", json={"enable": True}, timeout=10)
        time.sleep(2)
        requests.post(f"{RAPP_URL}/ue/airplane/off", json={"enable": False}, timeout=10)
        time.sleep(5)
        return True
    except Exception as e:
        print(f"    ✗ Airplane toggle error: {e}")
        return False

def run_iperf_test():
    """Run iperf test and return results"""
    try:
        resp = requests.post(
            f"{RAPP_URL}/ue/iperf",
            json={"duration": IPERF_DURATION, "bitrate": IPERF_BITRATE},
            timeout=IPERF_DURATION + 20
        )
        if resp.status_code == 200:
            data = resp.json()
            throughput = data.get('end', {}).get('streams', [{}])[0].get('udp', {}).get('bits_per_second', 0)
            throughput_mbps = throughput / 1_000_000
            jitter = data.get('end', {}).get('streams', [{}])[0].get('udp', {}).get('jitter_ms', 0)
            lost_percent = data.get('end', {}).get('streams', [{}])[0].get('udp', {}).get('lost_percent', 0)

            return {
                'throughput_mbps': throughput_mbps,
                'jitter_ms': jitter,
                'lost_percent': lost_percent
            }
        else:
            return None
    except Exception as e:
        print(f"    ✗ iperf error: {e}")
        return None

def fetch_thread_cpu():
    """Fetch thread CPU data during test"""
    print("  Fetching thread CPU data...")
    try:
        resp = requests.post(
            f"{RAPP_URL}/sideload/measure/thread_cpu",
            json={
                "instance_id": INSTANCE_ID,
                "duration": PERF_DURATION,
                "pgrep": "softmodem"
            },
            timeout=PERF_DURATION + 10
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    ✗ thread_cpu failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"    ✗ thread_cpu error: {e}")
        return None

def generate_plots(data, test_label, timestamp, iperf_result, ue_status):
    """Generate CPU profiling heatmaps only - per run"""
    # CPU usage heatmap
    df = pd.DataFrame(data['threads'])
    df['tid_numeric'] = pd.to_numeric(df['tid'])
    df = df.sort_values('tid_numeric')
    df['label'] = df['name'] + " (TID: " + df['tid'] + ")"
    df.set_index('label', inplace=True)
    heatmap_data = df[['min_cpu', 'avg_cpu', 'max_cpu']]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title(f'Thread CPU Usage - {test_label}\n{timestamp}')
    plt.xlabel('CPU Metrics')
    plt.ylabel('Thread Name & ID')
    plt.tight_layout()
    cpu_file = f'cpu_heatmap_{test_label}_{timestamp}.png'
    plt.savefig(cpu_file, dpi=150)
    print(f"  Saved {cpu_file}")
    plt.close()

    # Core affinity heatmap
    threads_with_cores = [t for t in data['threads'] if 'core_distribution' in t and t['core_distribution']]

    if threads_with_cores:
        system_max_cpu = data.get('system_max_cpu', 31)
        offline_cpus = set(data.get('offline_cpus', []))

        max_core_in_data = max(max(int(k) for k in t['core_distribution'].keys()) for t in threads_with_cores)
        max_core = max(max_core_in_data, system_max_cpu)

        matrix = []
        labels = []

        for thread in sorted(threads_with_cores, key=lambda x: float(x['tid'])):
            row = []
            core_dist = {int(k): v for k, v in thread['core_distribution'].items()}
            total_samples = sum(core_dist.values())

            for core in range(max_core + 1):
                count = core_dist.get(core, 0)
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                row.append(percentage)

            matrix.append(row)
            labels.append(f"{thread['name']} (TID: {thread['tid']})")

        df_cores = pd.DataFrame(matrix, index=labels, columns=[f"CPU{i}" for i in range(max_core + 1)])

        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_cores, annot=True, cmap='YlOrRd', fmt='.0f',
                    cbar_kws={'label': '% of samples'}, ax=ax)

        # Mark offline CPUs
        for cpu_id in offline_cpus:
            if cpu_id <= max_core:
                ax.add_patch(plt.Rectangle((cpu_id, 0), 1, len(df_cores),
                                           fill=False, edgecolor='blue',
                                           hatch='///', linewidth=2, zorder=10))

        if offline_cpus:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='none', edgecolor='blue',
                                    hatch='///', label='Offline CPU')]
            ax.legend(handles=legend_elements, loc='upper right')

        plt.title(f'Thread-to-Core Affinity - {test_label}\n{timestamp}\nOffline CPUs: {sorted(offline_cpus) if offline_cpus else "None"}')
        plt.xlabel('CPU Core')
        plt.ylabel('Thread Name & ID')
        plt.tight_layout()
        affinity_file = f'affinity_heatmap_{test_label}_{timestamp}.png'
        plt.savefig(affinity_file, dpi=150)
        print(f"  Saved {affinity_file}")
        plt.close()

def run_single_test(scenario, run_number, timestamp):
    """Execute single test run. If signal is missing, skip iperf and record 0."""
    print(f"\n  Run #{run_number + 1}/{scenario['runs']}")

    # 1. Toggle Airplane Mode
    toggle_airplane_mode()

    # 2. Check Attachment
    attached, ue_status = check_ue_status()

    # 3. Safe Signal Extraction
    signal = ue_status.get('signal') if ue_status else None

    # Retry once if attached but no signal
    if attached and signal is None:
        print("    ! Attached but no signal reporting. Waiting 5s...")
        time.sleep(5)
        _, ue_status = check_ue_status()
        signal = ue_status.get('signal') if ue_status else None

    # 4. "No Signal" Logic
    if not attached or signal is None:
        print("    ! NO SIGNAL DETECTED. Skipping traffic test.")
        return {
            'run': run_number + 1,
            'status': 'NO_SIGNAL',
            'throughput_mbps': 0.0,
            'jitter_ms': 0.0,
            'lost_percent': 0.0,
            'ptp_rms_ns': get_ptp_status(),
            'fh_connected': check_fh_connection(),
            'ue_rsrp': None,
            'ue_rsrq': None,
            'ue_sinr': None,
            'cpu_data': None
        }

    # 5. Normal Execution
    print(f"    UE: RSRP={signal.get('rsrp')} dBm, RSRQ={signal.get('rsrq')} dB, SINR={signal.get('sinr')} dB")

    ptp_rms = get_ptp_status()
    print(f"    PTP RMS: {ptp_rms} ns" if ptp_rms else "    PTP: N/A")

    fh_connected = check_fh_connection()

    print("    Starting iperf with CPU profiling...")
    iperf_result = run_iperf_test()

    time.sleep(1)
    cpu_data = fetch_thread_cpu()

    if iperf_result:
        print(f"    Throughput: {iperf_result['throughput_mbps']:.2f} Mbps")

    # Save per-run data
    if cpu_data:
        run_file = f"{scenario['test_id']}_run{run_number + 1}_{timestamp}.json"
        with open(run_file, 'w') as f:
            json.dump({
                'cpu_data': cpu_data,
                'iperf_result': iperf_result,
                'ue_status': ue_status
            }, f, indent=2)
        generate_plots(cpu_data, f"{scenario['test_id']}_R{run_number + 1}", timestamp, iperf_result, ue_status)

    return {
        'run': run_number + 1,
        'status': 'SUCCESS',
        'throughput_mbps': iperf_result['throughput_mbps'] if iperf_result else 0.0,
        'jitter_ms': iperf_result['jitter_ms'] if iperf_result else 0.0,
        'lost_percent': iperf_result['lost_percent'] if iperf_result else 0.0,
        'ptp_rms_ns': ptp_rms,
        'fh_connected': fh_connected,
        'ue_rsrp': signal.get('rsrp'),
        'ue_rsrq': signal.get('rsrq'),
        'ue_sinr': signal.get('sinr'),
        'cpu_data': cpu_data
    }

def run_test_scenario(scenario, timestamp):
    """Execute full test scenario with multiple runs"""
    test_id = scenario['test_id']
    print(f"\n{'='*70}")
    print(f"Test {test_id}: {scenario['oru'].upper()} O-RU")
    print(f"T1a_cp_dl: {scenario['T1a_cp_dl']}")
    print(f"{'='*70}")

    gnb_config = build_gnb_config(scenario)
    deployment_id = deploy_gnb(gnb_config)

    if not deployment_id:
        print(f"  ✗ Deployment failed, skipping {test_id}")
        return {'test_id': scenario['test_id'], 'oru': scenario['oru'].upper(), 'successful_runs': 0}

    print("  Waiting 30s for gNB stabilization...")
    time.sleep(30)

    run_results = []
    try:
        for run_num in range(scenario['runs']):
            result = run_single_test(scenario, run_num, timestamp)
            run_results.append(result)
            if run_num < scenario['runs'] - 1:
                time.sleep(3)
    finally:
        terminate_gnb(deployment_id)

    # Calculate statistics using numpy for safety
    valid_runs = [r for r in run_results if r['status'] in ['SUCCESS', 'NO_SIGNAL']]

    if valid_runs:
        def safe_mean(key):
            # Only count values that are not None
            vals = [r[key] for r in valid_runs if r.get(key) is not None]
            return np.mean(vals) if vals else None

        stats = {
            'test_id': scenario['test_id'],
            'oru': scenario['oru'].upper(),
            'T1a_cp_dl': scenario['T1a_cp_dl'],
            'Ta4': scenario['Ta4'],
            'total_runs': scenario['runs'],
            'successful_runs': len(valid_runs),
            'avg_throughput_mbps': np.mean([r['throughput_mbps'] for r in valid_runs]),
            'avg_jitter_ms': np.mean([r['jitter_ms'] for r in valid_runs]),
            'avg_loss_percent': np.mean([r['lost_percent'] for r in valid_runs]),
            'avg_ptp_rms_ns': safe_mean('ptp_rms_ns'),
            'avg_rsrp_dbm': safe_mean('ue_rsrp'),
            'avg_rsrq_db': safe_mean('ue_rsrq'),
            'avg_sinr_db': safe_mean('ue_sinr'),
            'fh_connected': any(r.get('fh_connected') for r in valid_runs),
            'raw_results': run_results
        }
    else:
        stats = {
            'test_id': scenario['test_id'],
            'oru': scenario['oru'].upper(),
            'successful_runs': 0,
            'raw_results': run_results
        }

    # Save results
    result_file = f'results_{scenario["test_id"]}_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats

def generate_summary_report(all_results, timestamp):
    """Generate comprehensive test report with Stacked CPU Breakdown"""

    # --- 1. Filter Valid Results ---
    valid_results = [r for r in all_results if r and 'test_id' in r]
    if not valid_results:
        print("No valid results to report.")
        return

    # --- 2. Prepare Data for Stacked CPU Chart ---
    # We need a DataFrame where Index=TestID, Columns=ThreadNames, Values=AvgCPU
    cpu_breakdown = {}

    for r in valid_results:
        test_id = r['test_id']
        cpu_breakdown[test_id] = {}

        # Aggregate thread usage across all runs for this test
        if r.get('raw_results'):
            thread_totals = {}
            run_count = 0

            for run in r['raw_results']:
                if run.get('cpu_data') and run['cpu_data'].get('threads'):
                    run_count += 1
                    for t in run['cpu_data']['threads']:
                        # Use thread name as key (strip numeric IDs if needed)
                        t_name = t['name']
                        # Simple sum for now, will divide by run_count later
                        thread_totals[t_name] = thread_totals.get(t_name, 0) + t['avg_cpu']

            # Calculate average per thread across runs
            if run_count > 0:
                for name, total in thread_totals.items():
                    cpu_breakdown[test_id][name] = total / run_count

    # Convert to DataFrame
    df_cpu = pd.DataFrame.from_dict(cpu_breakdown, orient='index').fillna(0)

    # Filter: Keep top 8 threads, group rest into "Others" to prevent clutter
    if not df_cpu.empty:
        top_threads = df_cpu.sum().nlargest(8).index
        df_cpu['Others'] = df_cpu.loc[:, ~df_cpu.columns.isin(top_threads)].sum(axis=1)
        df_cpu = df_cpu[list(top_threads) + ['Others']]

    # --- 3. Generate Visualizations ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    test_ids = [r['test_id'] for r in valid_results]
    colors = ['#1f77b4' if 'LITEON' in r['oru'] else '#ff7f0e' if 'PEGATRON' in r['oru'] else '#2ca02c' for r in valid_results]

    # Helper for standard bars
    def plot_metric(ax, data, title, ylabel, unit_label=""):
        bars = ax.bar(test_ids, data, color=colors, alpha=0.7)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, data)):
            is_no_signal = valid_results[i].get('avg_rsrp_dbm') is None
            if is_no_signal:
                ax.text(bar.get_x() + bar.get_width()/2., 0, 'No Signal',
                        ha='center', va='bottom', fontsize=8, color='red', rotation=90)
            elif val is not None:
                y_pos = val if val > 0 else 0
                if val < 0: y_pos = val
                ax.text(bar.get_x() + bar.get_width()/2., y_pos, f'{val:.1f}{unit_label}',
                        ha='center', va='bottom' if val>0 else 'top', fontsize=9)

    # Row 1
    plot_metric(fig.add_subplot(gs[0, 0]), [r.get('avg_throughput_mbps', 0) for r in valid_results], 'DL Throughput', 'Mbps')

    jitter_vals = [r.get('avg_jitter_ms', 0) if r.get('avg_jitter_ms') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[0, 1]), jitter_vals, 'Average Jitter', 'ms')

    loss_vals = [r.get('avg_loss_percent', 0) if r.get('avg_loss_percent') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[0, 2]), loss_vals, 'Packet Loss', '%', '%')

    # Row 2
    rsrp_vals = [r.get('avg_rsrp_dbm', 0) if r.get('avg_rsrp_dbm') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[1, 0]), rsrp_vals, 'Average RSRP', 'dBm')

    rsrq_vals = [r.get('avg_rsrq_db', 0) if r.get('avg_rsrq_db') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[1, 1]), rsrq_vals, 'Average RSRQ', 'dB')

    sinr_vals = [r.get('avg_sinr_db', 0) if r.get('avg_sinr_db') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[1, 2]), sinr_vals, 'Average SINR', 'dB')

    # Row 3 - METRIC 7: STACKED CPU BREAKDOWN (The new chart)
    ax7 = fig.add_subplot(gs[2, 1])
    if not df_cpu.empty:
        # Plot stacked bar
        df_cpu.plot(kind='bar', stacked=True, ax=ax7, colormap='tab20', width=0.7)
        ax7.set_title('CPU Usage Breakdown by Thread', fontweight='bold', fontsize=11)
        ax7.set_ylabel('CPU Usage (%)', fontweight='bold', fontsize=10)
        ax7.set_xlabel('Test ID', fontsize=10)
        ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=8)
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0)
    else:
        ax7.text(0.5, 0.5, 'No CPU Data', ha='center', va='center')

    # Row 3 - PTP
    ptp_vals = [r.get('avg_ptp_rms_ns', 0) if r.get('avg_ptp_rms_ns') is not None else 0 for r in valid_results]
    plot_metric(fig.add_subplot(gs[2, 0]), ptp_vals, 'PTP RMS', 'ns')

    # Row 3 - Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    table_data = []
    for r in valid_results:
        t1 = r.get('T1a_cp_dl', {})
        t4 = r.get('Ta4', {})
        table_data.append([
            r['test_id'],
            r['oru'],
            f"({t1.get('min')},{t1.get('max')})",
            f"({t4.get('min')},{t4.get('max')})"
        ])

    table = ax9.table(cellText=table_data, colLabels=['Test', 'O-RU', 'T1a_cp_dl', 'Ta4'], loc='center')
    table.scale(1, 1.5)

    plt.suptitle(f'O-RAN Timing Window Test Results - {timestamp}', fontsize=16, fontweight='bold')

    plot_file = f'consolidated_results_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Consolidated results plot saved to {plot_file}")
    plt.close()

    # Also save the breakdown to CSV for deep dive
    breakdown_file = f'cpu_breakdown_{timestamp}.csv'
    df_cpu.to_csv(breakdown_file)
    print(f"✓ CPU Breakdown saved to {breakdown_file}")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*70}")
    print(f"STARTING ITERATIVE gNB TIMING WINDOW TESTS")
    print(f"Timestamp: {timestamp}")
    print(f"Total Scenarios: {len(TEST_SCENARIOS)}")
    print(f"Total Runs: {sum(s['runs'] for s in TEST_SCENARIOS)}")
    print(f"{'='*70}")

    all_results = []

    for scenario in TEST_SCENARIOS:
        result = run_test_scenario(scenario, timestamp)
        all_results.append(result)

        print("\n  Waiting 10s before next scenario...")
        time.sleep(10)

    generate_summary_report(all_results, timestamp)

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
