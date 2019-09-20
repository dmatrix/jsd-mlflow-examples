from setuptools import setup
#
# To build the package simply `python setup.py` and `python setup.py sdist upload`
setup(name='imdbclassifier',
      version='0.6.6',
      description='MLflow modules for binary classification in Keras',
      url='http://github.com/dmatrix/jsd-mflow-examples/keras/imdbclassifier',
      author='Jules S. Damji',
      author_email='jules@databricks.com',
      license='None',
      packages=['imdbclassifier'],
      zip_safe=False)
