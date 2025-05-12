import setuptools
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="moonshot_prime",
    version="0.1.0", # Should match __version__ in __init__.py
    author="AetheCore", # Replace with your name/org
    author_email="AetheCoreContact@gamil.com", # Replace with your email
    description="Efficient prime detection engine for ultra-large numbers, optimized for Raspberry Pi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", # Optional: Add your project URL (e.g., GitHub repo)
    project_urls={ # Optional
        "Bug Tracker": "",
        "Source Code": "",
    },
    packages=setuptools.find_packages(), # Finds the moonshot_prime package
    # Or explicitly: packages=['moonshot_prime'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License", # Custom license specified
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Security :: Cryptography",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    include_package_data=False, # No extra data files needed for this simple package
    license_files=('LICENSE',),
) 