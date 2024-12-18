from setuptools import setup
from setuptools import find_packages

package_name = 'reasoner'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/msg', [
            'msg/DeicticSolution.msg',
            'msg/GDRNObject.msg',
            'msg/GDRNSolution.msg',
            'msg/HRICommand.msg'
        ]),
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
            'reasoner_node = reasoner.reasoner_node'
        ],
    },
)
