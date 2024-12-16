import sys, os, distutils.core, subprocess

dist = distutils.core.run_setup("../detectron2/setup.py")

for x in dist.install_requires:
    subprocess.run(['python', '-m', 'pip', 'install', x])

sys.path.insert(0, os.path.abspath('./detectron2'))