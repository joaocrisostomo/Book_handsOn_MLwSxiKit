from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='gpol',
    version='1.0.0',
    description='General Purpose Optimization Library (GPOL): a flexible and efficient multi-purpose optimization library in Python.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(exclude=('main',)),
    author='Illya Bakurov',
    author_email='ibakurov@isegi.unl.pt',
    keywords=['GPOL', 'Optimization'],
    url='',
    download_url='https://gitlab.com/ibakurov/general-purpose-optimization-library'
)

install_requires = [
    'appdirs==1.4.4',
    'filelock==3.0.12',
    'numpy==1.19.1',
    'pandas==1.1.5',
    'joblib==1.0.0',
    'scipy==1.6.0',
    #'torch==1.7.1',
]

package_data = {
    'gpol': ['utils/data/*.txt']
}

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires, package_data=package_data)
