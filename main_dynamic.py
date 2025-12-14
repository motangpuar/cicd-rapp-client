import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time

# Configuration
RAPP_URL = "http://192.168.8.35:5000"
NFO_URL = "http://192.168.8.35:8080"
INSTANCE_ID = "be0cc18b-1b8e-4b58-a2bc-9f4681ac6142"
IPERF_BITRATE = 200  # Mbps - target bitrate for all tests
IPERF_DURATION = 10
PERF_DURATION = 15

# Test matrix - iterative configurations
TEST_SCENARIOS = [
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
]

# TEST_SCENARIOS = [
#     {
#         "test_id": "P28",
#         "oru": "liteon",
#         "branch": "starlingx/liteon",
#         "T1a_cp_dl": {"min": 285, "max": 470},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 180},
#         "runs": 10
#     },
#     {
#         "test_id": "P29",
#         "oru": "liteon",
#         "branch": "starlingx/liteon",
#         "T1a_cp_dl": {"min": 285, "max": 550},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 280},
#         "runs": 10
#     },
#     {
#         "test_id": "P30",
#         "oru": "pegatron",
#         "branch": "starlingx/pegatron",
#         "T1a_cp_dl": {"min": 285, "max": 470},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 180},
#         "runs": 10
#     },
#     {
#         "test_id": "P31",
#         "oru": "pegatron",
#         "branch": "starlingx/pegatron",
#         "T1a_cp_dl": {"min": 285, "max": 550},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 280},
#         "runs": 10
#     },
#     {
#         "test_id": "P32",
#         "oru": "jura",
#         "branch": "starlingx/jura",
#         "T1a_cp_dl": {"min": 285, "max": 470},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 180},
#         "runs": 10
#     },
#     {
#         "test_id": "P33",
#         "oru": "jura",
#         "branch": "starlingx/jura",
#         "T1a_cp_dl": {"min": 285, "max": 550},
#         "T1a_cp_ul": {"min": 285, "max": 429},
#         "T1a_up": {"min": 125, "max": 350},
#         "Ta4": {"min": 110, "max": 280},
#         "runs": 10
#     }
# ]

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
        # rApp endpoint that forwards to NFO
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
        # rApp endpoint that forwards to NFO
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
            signal = status.get('signal', {})

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
            # Parse PTP offset to get RMS
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
        # Check gNB logs for FH connection
        resp = requests.get(f"{RAPP_URL}/gnb/logs", timeout=10)
        if resp.status_code == 200:
            logs = resp.text
            # Look for FH connection indicators
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
        # Use the working endpoint format from document 48
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
            print(f"    Response: {resp.text[:200]}")  # Debug output
            return None
    except Exception as e:
        print(f"    ✗ thread_cpu error: {e}")
        return None


def run_single_test(scenario, run_number, timestamp):
    """Execute single test run WITH profiling"""
    print(f"\n  Run #{run_number + 1}/{scenario['runs']}")

    # Toggle airplane mode
    toggle_airplane_mode()

    # Check UE attachment
    attached, ue_status = check_ue_status()
    if not attached:
        print("    ✗ UE not attached")
        return {
            'run': run_number + 1,
            'status': 'UE_NOT_ATTACHED'
        }

    signal = ue_status.get('signal', {})

    if signal == None:
        toggle_airplane_mode()
        time.sleep(5)

    print(f"    UE: RSRP={signal.get('rsrp')} dBm, RSRQ={signal.get('rsrq')} dB, SINR={signal.get('sinr')} dB")

    # Get PTP status
    ptp_rms = get_ptp_status()
    print(f"    PTP RMS: {ptp_rms} ns" if ptp_rms else "    PTP: N/A")

    # Check FH connection
    fh_connected = check_fh_connection()
    print(f"    FH Connected: {fh_connected}")

    # Start iperf in background and measure CPU during traffic
    print("    Starting iperf with CPU profiling...")
    iperf_result = run_iperf_test()

    # Measure thread CPU DURING traffic
    time.sleep(1)  # Let traffic stabilize
    cpu_data = fetch_thread_cpu()

    if iperf_result:
        print(f"    Throughput: {iperf_result['throughput_mbps']:.2f} Mbps")
        print(f"    Jitter: {iperf_result['jitter_ms']:.3f} ms")
        print(f"    Loss: {iperf_result['lost_percent']:.2f}%")

    # Save per-run data
    if cpu_data:
        run_file = f"{scenario['test_id']}_run{run_number + 1}_{timestamp}.json"
        with open(run_file, 'w') as f:
            json.dump({
                'cpu_data': cpu_data,
                'iperf_result': iperf_result,
                'ue_status': ue_status,
                'ptp_rms_ns': ptp_rms,
                'fh_connected': fh_connected
            }, f, indent=2)

        # Generate per-run heatmaps
        generate_plots(
            cpu_data,
            f"{scenario['test_id']}_R{run_number + 1}",
            timestamp,
            iperf_result,
            ue_status
        )

    return {
        'run': run_number + 1,
        'status': 'SUCCESS',
        'throughput_mbps': iperf_result['throughput_mbps'] if iperf_result else None,
        'jitter_ms': iperf_result['jitter_ms'] if iperf_result else None,
        'lost_percent': iperf_result['lost_percent'] if iperf_result else None,
        'ptp_rms_ns': ptp_rms,
        'fh_connected': fh_connected,
        'ue_rsrp': signal.get('rsrp'),
        'ue_rsrq': signal.get('rsrq'),
        'ue_sinr': signal.get('sinr'),
        'cpu_data': cpu_data  # Include for aggregation
    }

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

def run_test_scenario(scenario, timestamp):
    """Execute full test scenario with multiple runs"""
    test_id = scenario['test_id']
    oru = scenario['oru'].upper()

    print(f"\n{'='*70}")
    print(f"Test {test_id}: {oru} O-RU")
    print(f"T1a_cp_dl: ({scenario['T1a_cp_dl']['min']}, {scenario['T1a_cp_dl']['max']}) ns")
    print(f"Ta4: ({scenario['Ta4']['min']}, {scenario['Ta4']['max']}) ns")
    print(f"{'='*70}")

    # Build and deploy gNB
    gnb_config = build_gnb_config(scenario)
    deployment_id = deploy_gnb(gnb_config)

    if not deployment_id:
        print(f"  ✗ Deployment failed, skipping {test_id}")
        return None

    # Wait for gNB stabilization
    print("  Waiting 30s for gNB stabilization...")
    time.sleep(30)

    run_results = []
    try:
        for run_num in range(scenario['runs']):
            result = run_single_test(scenario, run_num, timestamp)  # Pass timestamp
            run_results.append(result)

            if run_num < scenario['runs'] - 1:
                time.sleep(3)

    finally:
        terminate_gnb(deployment_id)

    # Calculate statistics INCLUDING signal metrics
    successful_runs = [r for r in run_results if r['status'] == 'SUCCESS' and r['throughput_mbps'] is not None]

    if successful_runs:
        throughputs = [r['throughput_mbps'] for r in successful_runs]
        ptp_values = [r['ptp_rms_ns'] for r in successful_runs if r['ptp_rms_ns'] is not None]
        rsrp_values = [r['ue_rsrp'] for r in successful_runs if r['ue_rsrp'] is not None]
        rsrq_values = [r['ue_rsrq'] for r in successful_runs if r['ue_rsrq'] is not None]
        sinr_values = [r['ue_sinr'] for r in successful_runs if r['ue_sinr'] is not None]

        stats = {
            'test_id': scenario['test_id'],
            'oru': scenario['oru'].upper(),
            'T1a_cp_dl': scenario['T1a_cp_dl'],
            'Ta4': scenario['Ta4'],
            'total_runs': scenario['runs'],
            'successful_runs': len(successful_runs),
            'avg_throughput_mbps': sum(throughputs) / len(throughputs),
            'min_throughput_mbps': min(throughputs),
            'max_throughput_mbps': max(throughputs),
            'std_throughput_mbps': pd.Series(throughputs).std(),
            'avg_ptp_rms_ns': sum(ptp_values) / len(ptp_values) if ptp_values else None,
            'avg_rsrp_dbm': sum(rsrp_values) / len(rsrp_values) if rsrp_values else None,
            'avg_rsrq_db': sum(rsrq_values) / len(rsrq_values) if rsrq_values else None,
            'avg_sinr_db': sum(sinr_values) / len(sinr_values) if sinr_values else None,
            'fh_connected': any(r.get('fh_connected') for r in successful_runs),
            'raw_results': run_results
        }

    else:
        stats = {
            'test_id': scenario['test_id'],
            'oru': scenario['oru'].upper(),
            'total_runs': scenario['runs'],
            'successful_runs': 0,
            'raw_results': run_results
        }

    # Save scenario results
    result_file = f'results_{scenario["test_id"]}_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ Saved {result_file}")

    return stats



def generate_summary_report(all_results, timestamp):
    """Generate comprehensive test report"""

    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        if result:
            summary_data.append({
                'Test ID': result['test_id'],
                'O-RU': result['oru'],
                'T1a_cp_dl (ns)': result['T1a_cp_dl'],
                'Ta4 (ns)': result['Ta4'],
                'Runs': f"{result['successful_runs']}/{result['total_runs']}",
                'DL (Mbps)': f"{result['avg_throughput_mbps']:.2f}" if result.get('avg_throughput_mbps') else "N/A",
                'PTP RMS (ns)': f"{result['avg_ptp_rms_ns']:.2f}" if result.get('avg_ptp_rms_ns') else "N/A",
                'FH Connected': "Yes" if result.get('fh_connected') else "No"
            })

    df = pd.DataFrame(summary_data)

    # Print summary table
    print("\n" + "="*100)
    print("TEST SUMMARY")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Save summary
    summary_file = f'test_summary_{timestamp}.csv'
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Summary saved to {summary_file}")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    valid_results = [r for r in all_results if r and r.get('avg_throughput_mbps')]

    # Prepare data
    test_ids = [r['test_id'] for r in valid_results]
    colors = ['#1f77b4' if 'LITEON' in r['oru'] else '#ff7f0e' if 'PEGATRON' in r['oru'] else '#2ca02c' for r in valid_results]

    # Row 1: Performance Metrics
    # Throughput
    ax1 = fig.add_subplot(gs[0, 0])
    throughputs = [r['avg_throughput_mbps'] for r in valid_results]
    bars1 = ax1.bar(test_ids, throughputs, color=colors, alpha=0.7)
    ax1.set_ylabel('Throughput (Mbps)', fontsize=11, fontweight='bold')
    ax1.set_title('DL Throughput', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Jitter
    ax2 = fig.add_subplot(gs[0, 1])
    jitter_results = [r for r in valid_results if r.get('raw_results')]
    if jitter_results:
        jitter_avgs = []
        for r in jitter_results:
            jitters = [run.get('jitter_ms') for run in r['raw_results'] if run.get('jitter_ms')]
            jitter_avgs.append(sum(jitters)/len(jitters) if jitters else 0)
        bars2 = ax2.bar([r['test_id'] for r in jitter_results], jitter_avgs,
                       color=[colors[valid_results.index(r)] for r in jitter_results], alpha=0.7)
        ax2.set_ylabel('Jitter (ms)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Jitter', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, jitter_avgs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # Packet Loss
    ax3 = fig.add_subplot(gs[0, 2])
    loss_results = [r for r in valid_results if r.get('raw_results')]
    if loss_results:
        loss_avgs = []
        for r in loss_results:
            losses = [run.get('lost_percent') for run in r['raw_results'] if run.get('lost_percent') is not None]
            loss_avgs.append(sum(losses)/len(losses) if losses else 0)
        bars3 = ax3.bar([r['test_id'] for r in loss_results], loss_avgs,
                       color=[colors[valid_results.index(r)] for r in loss_results], alpha=0.7)
        ax3.set_ylabel('Packet Loss (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Average Packet Loss', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, loss_avgs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # Row 2: Signal Quality Metrics
    # RSRP
    ax4 = fig.add_subplot(gs[1, 0])
    rsrp_results = [r for r in valid_results if r.get('avg_rsrp_dbm')]
    if rsrp_results:
        rsrp_values = [r['avg_rsrp_dbm'] for r in rsrp_results]
        bars4 = ax4.bar([r['test_id'] for r in rsrp_results], rsrp_values,
                       color=[colors[valid_results.index(r)] for r in rsrp_results], alpha=0.7)
        ax4.set_ylabel('RSRP (dBm)', fontsize=11, fontweight='bold')
        ax4.set_title('Average RSRP', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars4, rsrp_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=9)

    # RSRQ
    ax5 = fig.add_subplot(gs[1, 1])
    rsrq_results = [r for r in valid_results if r.get('avg_rsrq_db')]
    if rsrq_results:
        rsrq_values = [r['avg_rsrq_db'] for r in rsrq_results]
        bars5 = ax5.bar([r['test_id'] for r in rsrq_results], rsrq_values,
                       color=[colors[valid_results.index(r)] for r in rsrq_results], alpha=0.7)
        ax5.set_ylabel('RSRQ (dB)', fontsize=11, fontweight='bold')
        ax5.set_title('Average RSRQ', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars5, rsrq_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=9)

    # SINR
    ax6 = fig.add_subplot(gs[1, 2])
    sinr_results = [r for r in valid_results if r.get('avg_sinr_db')]
    if sinr_results:
        sinr_values = [r['avg_sinr_db'] for r in sinr_results]
        bars6 = ax6.bar([r['test_id'] for r in sinr_results], sinr_values,
                       color=[colors[valid_results.index(r)] for r in sinr_results], alpha=0.7)
        ax6.set_ylabel('SINR (dB)', fontsize=11, fontweight='bold')
        ax6.set_title('Average SINR', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars6, sinr_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # Row 3: System Metrics
    # PTP RMS
    ax7 = fig.add_subplot(gs[2, 0])
    ptp_results = [r for r in valid_results if r.get('avg_ptp_rms_ns')]
    if ptp_results:
        ptp_values = [r['avg_ptp_rms_ns'] for r in ptp_results]
        bars7 = ax7.bar([r['test_id'] for r in ptp_results], ptp_values,
                       color=[colors[valid_results.index(r)] for r in ptp_results], alpha=0.7)
        ax7.set_ylabel('PTP RMS (ns)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Test ID', fontsize=11)
        ax7.set_title('PTP Synchronization', fontsize=12, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars7, ptp_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # Average CPU Usage
    ax8 = fig.add_subplot(gs[2, 1])
    cpu_results = [r for r in valid_results if r.get('raw_results')]
    if cpu_results:
        cpu_avgs = []
        for r in cpu_results:
            cpu_values = []
            for run in r['raw_results']:
                if run.get('cpu_data') and run['cpu_data'].get('threads'):
                    avg_cpu = sum(t['avg_cpu'] for t in run['cpu_data']['threads']) / len(run['cpu_data']['threads'])
                    cpu_values.append(avg_cpu)
            cpu_avgs.append(sum(cpu_values)/len(cpu_values) if cpu_values else 0)
        bars8 = ax8.bar([r['test_id'] for r in cpu_results], cpu_avgs,
                       color=[colors[valid_results.index(r)] for r in cpu_results], alpha=0.7)
        ax8.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Test ID', fontsize=11)
        ax8.set_title('Average CPU Usage', fontsize=12, fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars8, cpu_avgs):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 2, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # Parameter Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')

    param_table_data = []
    for r in valid_results:
        param_table_data.append([
            r['test_id'],
            r['oru'],
            f"({r['T1a_cp_dl']['min']},{r['T1a_cp_dl']['max']})",
            f"({r['Ta4']['min']},{r['Ta4']['max']})"
        ])

    table = ax9.table(cellText=param_table_data,
                     colLabels=['Test', 'O-RU', 'T1a_cp_dl', 'Ta4'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.2, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax9.set_title('Timing Parameters (ns)', fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='LiteON'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Pegatron'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='Jura')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.suptitle(f'O-RAN Timing Window Test Results - {timestamp}', fontsize=16, fontweight='bold')

    plot_file = f'consolidated_results_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Consolidated results plot saved to {plot_file}")
    plt.close()

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

        # Pause between scenarios
        print("\n  Waiting 10s before next scenario...")
        time.sleep(10)

    # Generate summary report
    generate_summary_report(all_results, timestamp)

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
