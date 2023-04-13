from setuptools import setup, find_packages

setup(
    name='littlebird',
    packages=find_packages(),
    version='latest',
    description='LittleBird: Efficient Faster & Longer Transformer for Question Answering',
    author='Teryn Jones',
    author_email='tkjones93@gmail.com',
    url='https://github.com/jwnz/littlebird', 
    install_requires=[
        'torch>=1.12.0'
    ],
    python_requires='>=3.8',
)
