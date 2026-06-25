"""
AWS deployment logger.

Writes to stdout (captured by Docker / CloudWatch) AND a rotating file
inside the container at /app/log/. Two handlers means you get live-tail
via `docker logs` on the EC2 host while also keeping a local copy.

Usage in app.py:
    from wasteDetection.logger.aws_logger import get_logger, log_aws_context, run_command
    logger = get_logger()
    log_aws_context(logger)
"""

import logging
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------------------------------
# EC2 instance metadata (IMDSv2)
# ---------------------------------------------------------------------------

def _imds(path: str, token: str, timeout: float = 1.0) -> str:
    """Return an IMDSv2 metadata field, or '' if not on EC2 / timed out."""
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return ""


def _get_imds_token(timeout: float = 1.0) -> str:
    """Obtain a short-lived IMDSv2 session token (valid 6 h)."""
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

_LOG_DIR = os.environ.get("APP_LOG_DIR", "/app/log")
_LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_LOG_BACKUP_COUNT = 5               # keep 5 rotated files


def get_logger(name: str = "wasteDetection") -> logging.Logger:
    """
    Return (or create) the named logger with two handlers:
      1. StreamHandler → stdout  (Docker captures this; forward to CloudWatch)
      2. RotatingFileHandler → /app/log/aws_<date>.log
    Calling this multiple times with the same name is safe — handlers are
    added only once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # 1. stdout (picked up by `docker logs` and CloudWatch if agent is installed)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    # 2. rotating file inside the container
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        log_file = os.path.join(
            _LOG_DIR,
            f"aws_{datetime.now().strftime('%Y-%m-%d')}.log",
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log file: %s", log_file)
    except OSError as exc:
        logger.warning("Could not open log file in %s: %s", _LOG_DIR, exc)

    return logger


# ---------------------------------------------------------------------------
# Startup context
# ---------------------------------------------------------------------------

def log_aws_context(logger: logging.Logger) -> None:
    """
    Log AWS/container context once at startup so every log session begins
    with a clear fingerprint of where the app is running.
    """
    token = _get_imds_token()
    instance_id   = _imds("instance-id", token)            or "not-on-ec2"
    instance_type = _imds("instance-type", token)          or "unknown"
    az            = _imds("placement/availability-zone", token) or "unknown"
    public_ip     = _imds("public-ipv4", token)            or "unknown"

    region       = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "unknown")
    container_id = os.environ.get("HOSTNAME", "unknown")   # Docker sets HOSTNAME to short container ID

    logger.info("=" * 50)
    logger.info("AWS DEPLOYMENT CONTEXT")
    logger.info("  EC2 Instance ID   : %s", instance_id)
    logger.info("  EC2 Instance Type : %s", instance_type)
    logger.info("  Availability Zone : %s", az)
    logger.info("  Public IP         : %s", public_ip)
    logger.info("  AWS Region        : %s", region)
    logger.info("  Container ID      : %s", container_id)
    logger.info("  Python            : %s", sys.version.split()[0])
    logger.info("  CWD               : %s", os.getcwd())
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Subprocess helper (replaces os.system)
# ---------------------------------------------------------------------------

def run_command(cmd: str, logger: logging.Logger) -> int:
    """
    Run a shell command and stream every output line through the logger.
    Returns the process exit code.

    Replaces bare os.system() calls so that YOLOv5 detect.py output
    and any errors are visible in Docker logs / CloudWatch instead of
    being silently discarded.
    """
    logger.info("CMD START: %s", cmd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if line.strip():
                logger.info("[cmd] %s", line)
        if proc.returncode != 0:
            logger.error("CMD FAILED (exit %d): %s", proc.returncode, cmd)
        else:
            logger.info("CMD OK (exit 0): %s", cmd)
        return proc.returncode
    except Exception:
        logger.exception("CMD EXCEPTION: %s", cmd)
        return -1
