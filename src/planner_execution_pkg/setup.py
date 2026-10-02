from setuptools import find_packages, setup

package_name = 'planner_execution_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='miguel-calle',
    maintainer_email='miguel-calle@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'compare_planners = planner_execution_pkg.dual_planner_node:main',
            'execute_planner = planner_execution_pkg.dual_planner_node:main',
            'fine_approach = planner_execution_pkg.fine_approach_node:main',
            'pick_place_cycle = planner_execution_pkg.pick_place_cycle_node:main',
        ],
    },
)