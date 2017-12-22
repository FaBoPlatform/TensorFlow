pip3 install --upgrade jupyter

########################################
# Jupyter setting
########################################
mkdir -p /home/ubuntu/notebooks
# 起動方法
# env PASSWORD=gclue jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000

# 設定ファイル用意
# /root/.jupyter/jupyter_notebook_config.py
#import os
#from IPython.lib import passwd
#
#c.NotebookApp.ip = '*'
#c.NotebookApp.notebook_dir = u'/notebooks/'
#c.NotebookApp.port = int(os.getenv('PORT', 8888))
#c.NotebookApp.open_browser = False
#c.MultiKernelManager.default_kernel_name = 'python2'
#
## sets a password if PASSWORD is set in the environment
#if 'PASSWORD' in os.environ:
#  c.NotebookApp.password = passwd(os.environ['PASSWORD'])
#  del os.environ['PASSWORD']

jupyter notebook --generate-config --allow-root

echo -e "import os\n\
from IPython.lib import passwd\n\
\n\
c.NotebookApp.ip = '*'\n\
c.NotebookApp.notebook_dir = u'/home/ubuntu/notebooks/'\n\
c.NotebookApp.port = int(os.getenv('PORT', 8888))\n\
#c.NotebookApp.allow_origin='http://localhost:8888'\n\
c.NotebookApp.open_browser = False\n\
c.MultiKernelManager.default_kernel_name = 'python3'\n\
\n\
# sets a password if PASSWORD is set in the environment\n\
if 'PASSWORD' in os.environ:\n\
  c.NotebookApp.password = passwd(os.environ['PASSWORD'])\n\
  del os.environ['PASSWORD']\n"\
>> /root/.jupyter/jupyter_notebook_config.py


########################################
# Jupyter自動起動
########################################
# パスワード：mypassword
cat <<EOF> /etc/init.d/jupyterd
#!/bin/sh
### BEGIN INIT INFO
# Provides:         jupyterd
# Required-Start:   $remote_fs $syslog
# Required-Stop:    $remote_fs $syslog
# Default-Start:    2 3 4 5
# Default-Stop:	    0 1 6
# Short-Description: Jupyter launcher
### END INIT INFO

# Launch Jupyter
env PASSWORD=mypassword jupyter notebook --allow-root --NotebookApp.iopub_data_rate_limit=10000000
EOF

chmod 755 /etc/init.d/jupyterd
update-rc.d jupyterd defaults

