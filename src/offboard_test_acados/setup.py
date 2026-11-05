import glob

from setuptools import find_packages, setup

package_name = 'offboard_test_acados'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob.glob('config/**/*.yaml', recursive=True)),
        ('share/' + package_name + '/worlds', glob.glob('worlds/**/*.sdf', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clear',
    maintainer_email='chasequarkko@gmail.com',
    description='Payload-adaptive NMPC with a velocity-mass MHE for a gripper quadrotor (mainline)',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'acados_nmpc_node = offboard_test_acados.acados_nmpc_node:main',
            'mhe_node = offboard_test_acados.mhe.node:main',
            'proximity_gripper_node = offboard_test_acados.gripper.proximity_gripper_node:main',
        ],
    },
)
