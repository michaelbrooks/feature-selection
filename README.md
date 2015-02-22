# Feature Selection Tests

## Setup

Using VirtualBox and Vagrant (recommended, especially on Windows):

1. Install VirtualBox and Vagrant on your machine.
2. Clone the repo.
3. Run `vagrant up` in the repo directory. This installs all the dependencies.
4. SSH into the VM. Run `workon feature-selection` and `python test.py`.

If you aren't using VirtualBox and Vagrant:

1. You need to have Python 2.7 with numpy and scipy installed.
2. Run `pip install -r requirements.txt` to install the 
   dependencies specified in the `requirements.txt file.
