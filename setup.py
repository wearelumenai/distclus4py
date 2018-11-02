from setuptools import setup, find_packages

requires=[
    'cffi>=1.0.0',
    'numpy>=1.6',
    'scipy>=0.13.0'
]

tests_require=[
    'pytest',
    'pytest-cov'
]

setup(
    name='distclus',
    version='0.1',
    description='distclus golang facade',
    author='lumenai',
    author_email='',
    url='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'test': tests_require,
    },
    install_requires=requires,
)