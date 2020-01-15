from setuptools import find_packages, setup


def get_requirements():
    with open('requirements.txt') as fh:
        requirements = fh.read().splitlines()
    return requirements


def get_developer_requirements():
    with open('requirements-dev.txt') as fh:
        requirements = fh.read().splitlines()
    requirements = [x for x in requirements if not x.startswith('Cython')]
    return requirements


setup(
    name='repro_vision',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={
        'dev': get_developer_requirements(),
    },
    entry_points='''
        [console_scripts]
        repro_vision=repro_vision.commands:main
    '''
)
