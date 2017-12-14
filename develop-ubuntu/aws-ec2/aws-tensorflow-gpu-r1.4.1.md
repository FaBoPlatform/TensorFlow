# LAUNCH
- make $HOME/notebook directory
> - mkdir /home/ubuntu/notebooks
- pull docker image
> - nvidia-docker pull naisy/aws-tensorflow-gpu-x86_64:r1.4.1
- make docker container and start. (change mypassword for jupyter login)
> - nvidia-docker run -itd -v /home/ubuntu/notebooks:/notebooks -e "PASSWORD=mypassword" -p 6006:6006 -p 8888:8888 naisy/aws-tensorflow-gpu-x86_64:r1.4.1 /bin/bash -c "jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000"

# NEED
- jupyter setting, edit for YOUR DOMAIN
> - nvidia-docker ps -a
> - nvidia-docker exec -it CONTAINER_ID /bin/bash
>- vi /root/.jupyter/jupyter_notebook_config.py
c.NotebookApp.allow_origin='YOUR DOMAIN:8888'

### SEE ALSO:
* [naisy/aws-tensorflow-gpu-x86_64 (Docker Hub)](https://hub.docker.com/r/naisy/aws-tensorflow-gpu-x86_64/)
