import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
user = os.getenv("DB_USER", "postgres")
password = os.getenv("DB_PASSWORD", "")

print("=== Quick connectivity test ===")
print(f"Host: {host}")
print(f"Port: {port}")
print()

# Test 1: TCP connectivity
print("Test 1: TCP connectivity (nc -z)...")
r = subprocess.run(["nc", "-z", str(host), str(port)], capture_output=True, text=True, timeout=5)
print(f"  Exit code: {r.returncode}")
if r.returncode != 0:
    print(f"  stderr: {r.stderr.strip()}")
    print("  -> Port is NOT reachable.")
else:
    print("  -> Port is reachable.")

print()

# Test 2: psql connection
print("Test 2: psql connection to database 'postgres'...")
env = os.environ.copy()
if password:
    env["PGPASSWORD"] = password

cmd = [
    "psql",
    "-h", str(host),
    "-p", str(port),
    "-U", str(user),
    "-d", "postgres",
    "-c", "SELECT 1",
]
r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
print(f"  Exit code: {r2.returncode}")
if r2.returncode != 0:
    print(f"  stderr: {r2.stderr.strip()[:500]}")
else:
    print(f"  stdout: {r2.stdout.strip()}")
    print("  -> Connection successful!")