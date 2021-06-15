from setuptools import find_packages, setup

# with open('README.md', 'r') as fh:
#     long_description = fh.read()

setup(
    name='plainedft',
    version='0.0.1',
    description='Simple plane wave density funtional theory code.',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    # url='https://gitlab.com/wangenau/variational_mesh',
    author='Wanja Schulze',
    author_email='wangenau@protonmail.com',
    license='APACHE2.0',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'scipy'],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False
)
