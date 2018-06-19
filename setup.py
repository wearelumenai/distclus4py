from setuptools import setup, find_packages

setup(
    name='distclus_bind',
    version='0.0',
    description='distclus golang bind',
    author='lumenai',
    author_email='',
    url='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'cffi>=1.0.0',
        'numpy>=1.6',
        'scipy>=0.13.0'
    ],
)