# dalle-mini-api
codebase that deploys Dalle-Mini as an API

# Instructions to run and install
* create p3.2xlarge EC2 instance
* Use Conda AMI, other AMIs do not have nvidia-smi installed by default
* ssh
* `sudo su`
* `cd gpu/src/`
* `pip install -r requirements.txt`
* run `python app.py`
* use `gpu/request_test.py` to test API POST request
