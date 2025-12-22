from setuptools import find_packages, setup

setup(
    name='ExoCAT',
    packages=find_packages(include=['ExoCAT']),
    version='0.0.1',
    description='Codes to analyse JWST/MIRI high-contrast imaging data for exoplanets : MIRI coronagraphs + MIRI/MRS HCI and cross-correlation',
    author='Mathilde Mâlin',
    #license='MIT',
)