import sys
import os

try:
    print("Verifying syntax of app_licensed.py...")
    with open('app_licensed.py', 'r', encoding='utf-8') as f:
        compile(f.read(), 'app_licensed.py', 'exec')
    print("Syntax OK")
except Exception as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)
