# reasoner/setup.py
from setuptools import setup

package_name = 'reasoner'

setup(
    name=package_name,
    version='0.0.1',
    packages=['reasoner'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Simon Dratva',
    maintainer_email='dratvsim@fel.cvut.cz',
    description='',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reasoner_node = reasoner.reasoner_node:main',
        ],
    },
)