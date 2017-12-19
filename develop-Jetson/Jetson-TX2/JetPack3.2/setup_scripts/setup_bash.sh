########################################
# .bashrc (ubuntu/root両方)
########################################
# sed
# escape characters \'$.*/[]^
# 1. Write the regex between single quotes.
# 2. \ -> \\
# 3. ' -> '\''
# 4. Put a backslash before $.*/[]^ and only those characters.

####################
# ubuntu user
####################
#-    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
#+    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\[\033[01;32m\]\h\[\033[00;00m\]:\[\033[01;35m\]\w\[\033[00m\]\$ '
#-    alias ls='ls --color=auto'
#+    alias ls='ls -asiF --color=auto'
sed -i 's/PS1='\''\${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;32m\\\]\\u@\\h\\\[\\033\[00m\\\]:\\\[\\033\[01;34m\\\]\\w\\\[\\033\[00m\\\]\\\$ '\''/PS1='\''\${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;32m\\\]\\u@\\\[\\033\[01;32m\\\]\\h\\\[\\033\[00;00m\\\]:\\\[\\033\[01;35m\\\]\\w\\\[\\033\[00m\\\]\\\$ '\''/g' /home/ubuntu/.bashrc
sed -i 's/alias ls='\''ls --color=auto'\''/alias ls='\''ls -asiF --color=auto'\''/g' /home/ubuntu/.bashrc

cat <<EOF>> /home/ubuntu/.bashrc

export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:
export __GL_PERFMON_MODE=1
export LANG="en_US.UTF-8"
export LC_ALL=$LANG
export LC_CTYPE=$LANG
EOF

####################
# root user
####################
#-    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
#+    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;37m\]\u@\[\033[01;32m\]\h\[\033[00;00m\]:\[\033[01;35m\]\w\[\033[00m\]\$ '
#-    alias ls='ls --color=auto'
#+    alias ls='ls -asiF --color=auto'
#-    xterm-color) color_prompt=yes;;
#+    xterm-color|*-256color) color_prompt=yes;;
sed -i 's/PS1='\''\${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;32m\\\]\\u@\\h\\\[\\033\[00m\\\]:\\\[\\033\[01;34m\\\]\\w\\\[\\033\[00m\\\]\\\$ '\''/PS1='\''\${debian_chroot:+(\$debian_chroot)}\\\[\\033\[01;37m\\\]\\u@\\\[\\033\[01;32m\\\]\\h\\\[\\033\[00;00m\\\]:\\\[\\033\[01;35m\\\]\\w\\\[\\033\[00m\\\]\\\$ '\''/g' /root/.bashrc
sed -i 's/alias ls='\''ls --color=auto'\''/alias ls='\''ls -asiF --color=auto'\''/g' /root/.bashrc
sed -i 's/xterm-color) color_prompt=yes;;/xterm-color|\*-256color) color_prompt=yes;;/g' /root/.bashrc

cat <<EOF>> /root/.bashrc

export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:
export __GL_PERFMON_MODE=1
export LANG="en_US.UTF-8"
export LC_ALL=$LANG
export LC_CTYPE=$LANG
EOF
