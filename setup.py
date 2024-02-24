import os
from setuptools import setup, find_packages
from setuptools.command.install import install

class AddLogsDir(install):
    def run(self):
        # Add logs/ directory for experiments
        os.makedirs('logs', exist_ok=True)
        # Call the standard install
        install.run(self)

setup(
    name="EpiK-Eval",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    cmdclass={
        'install': AddLogsDir,
    },
)