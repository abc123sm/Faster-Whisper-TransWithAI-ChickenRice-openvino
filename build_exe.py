import PyInstaller.__main__
import os
import shutil

if __name__ == '__main__':
    print("Starting PyInstaller build...")
    
    # Run PyInstaller
    PyInstaller.__main__.run([
        'vino.spec',
        '--clean',
        '--noconfirm',
        '--distpath', 'dist_release',
        '--workpath', 'build_release',
    ])
    
    print("Build completed.")
