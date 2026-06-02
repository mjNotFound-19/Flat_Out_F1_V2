#!/usr/bin/env python3
"""
Setup script for the F1 Telemetry Collector.
Installs dependencies and configures auto-start on boot.

Supports:
  - macOS (launchd)
  - Linux (systemd)
  - Windows (Task Scheduler)
"""

import os
import sys
import platform
import subprocess
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
COLLECTOR_SCRIPT = SCRIPT_DIR / "f1_collector.py"
PYTHON = sys.executable

# -- Determine data directory --
# Check if f1_collector.py has a custom base_dir configured,
# otherwise default to ~/f1_telemetry_data
def _get_data_dir() -> Path:
    """Read base_dir from CollectorConfig if available, else use default."""
    try:
        # Import the config to respect any user customizations
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "f1_collector", str(COLLECTOR_SCRIPT)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return Path(mod.CollectorConfig().base_dir)
    except Exception:
        return Path.home() / "f1_telemetry_data"

DATA_DIR = _get_data_dir()


def install_dependencies():
    """Install required Python packages."""
    print("\n[INSTALL] Installing dependencies...")
    deps = ["fastf1", "pandas", "pyarrow"]
    for dep in deps:
        print(f"  Installing {dep}...")
        subprocess.run(
            [PYTHON, "-m", "pip", "install", dep, "--quiet"],
            check=False
        )
    print("  [OK] Dependencies installed\n")


def setup_macos():
    """Create a launchd plist for macOS auto-start."""
    plist_name = "com.flatoutf1.collector"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{plist_name}.plist"
    log_path = DATA_DIR / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    plist_content = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>Label</key>
        <string>{plist_name}</string>

        <key>ProgramArguments</key>
        <array>
            <string>{PYTHON}</string>
            <string>{COLLECTOR_SCRIPT}</string>
        </array>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <dict>
            <key>SuccessfulExit</key>
            <false/>
        </dict>

        <key>StandardOutPath</key>
        <string>{log_path / "launchd_stdout.log"}</string>

        <key>StandardErrorPath</key>
        <string>{log_path / "launchd_stderr.log"}</string>

        <key>WorkingDirectory</key>
        <string>{SCRIPT_DIR}</string>

        <key>ThrottleInterval</key>
        <integer>60</integer>

        <key>ProcessType</key>
        <string>Background</string>

        <key>LowPriorityBackgroundIO</key>
        <true/>
    </dict>
    </plist>
    """)

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    print(f"  [OK] Created launchd plist: {plist_path}")

    # Load the agent
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    print(f"  [OK] Loaded launchd agent")

    print(f"""
  +======================================================+
  |           macOS Setup Complete!                      |
  +======================================================+
  |                                                      |
  |  The collector will now start automatically           |
  |  whenever you log in.                                |
  |                                                      |
  |  Commands:                                           |
  |    Status:  python f1_collector.py --status           |
  |    Stop:    launchctl unload {plist_path.name:>20}    |
  |    Start:   launchctl load {plist_path.name:>22}      |
  |    Logs:    tail -f ~/f1_telemetry_data/logs/*.log   |
  |                                                      |
  +======================================================+
    """)


def setup_linux():
    """Create a systemd user service for Linux auto-start."""
    service_name = "f1-collector"
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_path = service_dir / f"{service_name}.service"

    service_content = textwrap.dedent(f"""\
    [Unit]
    Description=Flat Out F1 v2 - Telemetry Data Collector
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    ExecStart={PYTHON} {COLLECTOR_SCRIPT}
    WorkingDirectory={SCRIPT_DIR}
    Restart=on-failure
    RestartSec=60
    Environment=PYTHONUNBUFFERED=1

    # Resource limits - be a good citizen
    CPUQuota=25%
    MemoryMax=512M
    IOWeight=50

    [Install]
    WantedBy=default.target
    """)

    service_dir.mkdir(parents=True, exist_ok=True)
    service_path.write_text(service_content)
    print(f"  [OK] Created systemd service: {service_path}")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
    subprocess.run(["systemctl", "--user", "start", service_name], check=True)
    # Enable lingering so the service runs even when not logged in
    subprocess.run(["loginctl", "enable-linger"], capture_output=True)

    print(f"""
  +======================================================+
  |           Linux Setup Complete!                      |
  +======================================================+
  |                                                      |
  |  The collector runs as a systemd user service.       |
  |                                                      |
  |  Commands:                                           |
  |    Status:  systemctl --user status {service_name}    |
  |    Stop:    systemctl --user stop {service_name}      |
  |    Start:   systemctl --user start {service_name}     |
  |    Logs:    journalctl --user -u {service_name} -f    |
  |    Disable: systemctl --user disable {service_name}   |
  |                                                      |
  +======================================================+
    """)


def setup_windows():
    """Create a Windows Task Scheduler task for auto-start."""
    task_name = "FlatOutF1Collector"
    vbs_path = SCRIPT_DIR / "start_collector.vbs"
    bat_path = SCRIPT_DIR / "start_collector.bat"

    # Create batch file
    bat_content = textwrap.dedent(f"""\
    @echo off
    cd /d "{SCRIPT_DIR}"
    "{PYTHON}" "{COLLECTOR_SCRIPT}"
    """)
    bat_path.write_text(bat_content)

    # Create VBS wrapper to run without a visible console window
    vbs_content = textwrap.dedent(f"""\
    Set WshShell = CreateObject("WScript.Shell")
    WshShell.Run chr(34) & "{bat_path}" & chr(34), 0
    Set WshShell = Nothing
    """)
    vbs_path.write_text(vbs_content)

    # Create scheduled task via schtasks
    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'wscript.exe "{vbs_path}"',
        "/sc", "ONLOGON",
        "/rl", "LIMITED",
        "/f",  # force overwrite if exists
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  [OK] Created scheduled task: {task_name}")
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Could not create task automatically.")
        print(f"    Run as administrator or create manually:")
        print(f"    schtasks /create /tn {task_name} /tr \"wscript.exe {vbs_path}\" /sc ONLOGON")

    print(f"""
  +======================================================+
  |           Windows Setup Complete!                    |
  +======================================================+
  |                                                      |
  |  The collector runs via Task Scheduler at login.     |
  |                                                      |
  |  Commands:                                           |
  |    Status:  python f1_collector.py --status           |
  |    Stop:    python f1_collector.py --stop             |
  |    Manual:  python f1_collector.py                    |
  |    Remove:  schtasks /delete /tn {task_name}          |
  |                                                      |
  +======================================================+
    """)


def main():
    system = platform.system()

    print("+======================================================+")
    print("|      FLAT OUT F1 v2 - Collector Setup               |")
    print("+======================================================+")
    print(f"|  Platform:  {system:<40} |")
    print(f"|  Python:    {PYTHON:<40} |")
    print(f"|  Script:    {str(COLLECTOR_SCRIPT):<40} |")
    print(f"|  Data dir:  {str(DATA_DIR):<40} |")
    print("+======================================================+")

    # 1. Install deps
    install_dependencies()

    # 2. Create data directory structure
    print("[DIRS] Creating directory structure...")
    for subdir in ["raw/openf1", "processed", "cache/fastf1_cache",
                    "logs", "live_recordings"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print("  [OK] Directories created\n")

    # 3. Platform-specific auto-start
    print(f"[CONFIG]  Configuring auto-start for {system}...")
    if system == "Darwin":
        setup_macos()
    elif system == "Linux":
        setup_linux()
    elif system == "Windows":
        setup_windows()
    else:
        print(f"  [WARN] Unknown platform: {system}")
        print(f"  Run manually: python {COLLECTOR_SCRIPT}")

    # 4. Run initial backfill
    print("\n[SYNC] Running initial backfill of recent sessions...")
    print("   (This may take a few minutes)\n")
    subprocess.run(
        [PYTHON, str(COLLECTOR_SCRIPT), "--backfill", "--days", "14"],
        check=False
    )

    print("\n[DONE] Setup complete! The collector is now running.")


if __name__ == "__main__":
    main()
