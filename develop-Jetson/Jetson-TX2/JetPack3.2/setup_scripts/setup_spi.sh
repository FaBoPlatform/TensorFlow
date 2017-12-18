########################################
# SPIDev有効化
########################################
# 参考
#  * [\*C] https://elinux.org/Jetson/TX1_SPI#Installing_SPIdev_Kernel_Module
#  * [\*D] https://elinux.org/Jetson/TX1_SPI#Installing_DTC_Tool
#  * [\*B] https://devtalk.nvidia.com/default/topic/1008929/jetson-tx2/enabling-spi-and-spidev-on-the-tx2/
#  * [\*A] https://devtalk.nvidia.com/default/topic/1023007/how-to-use-uart0-as-normal-uart-port-on-r28-1-/?offset=12
#  * [\*E] https://devtalk.nvidia.com/default/topic/1020708/method-to-modify-use-different-device-tree-in-r28-1/

# ソースコードのファイル名、展開後のディレクトリ名は毎回変わっているので確認すること
SCRIPT_DIR=$(cd $(dirname $0); pwd)

mkdir /compile
cd /compile
wget --no-check-certificate https://developer.nvidia.com/embedded/dlc/l4t-sources-28-2 -O sources_r28.2.tbz2
tar -xvf sources_r28.2.tbz2
cd public_release

tar -xvf kernel_src.tbz2
cd kernel/kernel-4.4

zcat /proc/config.gz > .config
# less .configで内容を確認してから実行すること
sed -i 's/CONFIG_LOCALVERSION=""/CONFIG_LOCALVERSION="-tegra"/' .config
sed -i 's/# CONFIG_SPI_SPIDEV is not set/CONFIG_SPI_SPIDEV=m/' .config

make prepare
make modules_prepare
make M=drivers/spi/

cp drivers/spi/spidev.ko /lib/modules/$(uname -r)/kernel/drivers
depmod

## DTC Tool [*D]
apt-get update
apt-get install -y device-tree-compiler

## SPI Configuration [*B]
cd /boot/dtb
# check extlinux.conf
cat /boot/extlinux/extlinux.conf
## get DTB from image. [*E] https://devtalk.nvidia.com/default/topic/1020708/method-to-modify-use-different-device-tree-in-r28-1/
dd if=/dev/mmcblk0p25 of=test.dtb
dtc -I dtb -O dts -o test.dts test.dtb
rm test.dtb

# backup
cp tegra186-quill-p3310-1000-c03-00-base.dtb tegra186-quill-p3310-1000-c03-00-base.dtb.org
# decompile
dtc -I dtb -O dts -o tegra186-quill-p3310-1000-c03-00-base.dts tegra186-quill-p3310-1000-c03-00-base.dtb
# patch
patch -u /boot/dtb/tegra186-quill-p3310-1000-c03-00-base.dts < $SCRIPT_DIR/spi.patch 
# compile
dtc -I dts -O dtb -o tegra186-quill-p3310-1000-c03-00-base.dtb tegra186-quill-p3310-1000-c03-00-base.dts
## dtb into /dev/mmcblk0p15 [*A]
# JetPack 3.2は/dev/mmcblk0p25
# JetPack 3.1は/dev/mmcblk0p15
dd if=/boot/dtb/tegra186-quill-p3310-1000-c03-00-base.dtb of=/dev/mmcblk0p25

#reboot

#lsmod
##Module                  Size  Used by
##fuse                   82192  2
##bcmdhd               7441995  0
##spidev                  9920  0
##pci_tegra              61290  0
##bluedroid_pm           11195  0

#ls /dev/spidev3.*
#/dev/spidev3.0  /dev/spidev3.1
