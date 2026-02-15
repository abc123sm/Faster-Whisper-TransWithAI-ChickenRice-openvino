#!/usr/bin/env python3
"""
Standalone inference script with custom VAD injection
This can be run directly from the project root without installation
"""

import sys
import os

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from faster_whisper_transwithai_chickenrice.infer import main

if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))
    main()