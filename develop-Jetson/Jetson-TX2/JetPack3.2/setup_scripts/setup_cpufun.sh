########################################
# CPUファン自動起動
########################################
cat <<EOF> /etc/init.d/cpufan
#!/bin/sh
### BEGIN INIT INFO
# Provides:         cpufan
# Required-Start:   $remote_fs $syslog
# Required-Stop:    $remote_fs $syslog
# Default-Start:    2 3 4 5
# Default-Stop:	    0 1 6
# Short-Description: CPU Fan launcher
### END INIT INFO

# Launch CPU Fan
sh -c 'echo 255 > /sys/kernel/debug/tegra_fan/target_pwm'
EOF

chmod 755 /etc/init.d/cpufan
update-rc.d cpufan defaults
