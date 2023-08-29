from setuptools import setup, find_packages

setup(name='pde_datasets',
      version='0.1',
      description='Collection of PDE Problems',
      url='https://github.com/FabricioArendTorres/burgers_analytical_1d_2d',
      author='Fabricio Arend Torres',
      author_email='fabricio.arendtorres@unibas.ch',
      license='MIT',
      install_requires=[
          "ipdb",
          "ipython",
          "matplotlib==3.5.1",
          "numba",
          "numpy",
          "pandas==1.4.2",
          "plotly==5.6.0",
          "scikit-learn==1.1.1",
          "scipy==1.8.1",
          "seaborn==0.11.2",
          "snappy",
          "sympy==1.10.1",
          "tqdm",
          "pylatexenc"
      ],
      zip_safe=False)
