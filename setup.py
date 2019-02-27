from setuptools import setup


setup(name='MNIST',
      packages=['MNIST'],
      version='0.0.1',
      entry_points={
            'console_scripts': ['MNIST=main.cmd:main']
      }
)
