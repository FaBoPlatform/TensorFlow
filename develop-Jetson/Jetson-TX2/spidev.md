# Jetson TX2 SPIdevを有効化

###### メモ
* Jetson TX2はtegra186
* Kernel Module compile
* DTB decompile
* DTS編集
* DTB compile
* /dev/mmcblk0p15に書き込む
* reboot
* spi@3240000はspi@c260000にマッピングされているので同一PINになる。

check cmmands
```
lsmod
ls /proc/device-tree/ | grep spi
cat /proc/modules
ls /dev/spi*
```

## Install SPIdev module [*C]
```
wget --no-check-certificate https://developer.nvidia.com/embedded/dlc/l4t-sources-28-1 -O sources_r28.1.tbz2
tar -xvf sources_r28.1.tbz2
cd sources

tar -xvf kernel_src-tx2.tbz2
cd kernel/kernel-4.4

zcat /proc/config.gz > .config
# less .configで内容を確認してから実行すること
sed -i 's/# CONFIG_SPI_SPIDEV is not set/CONFIG_SPI_SPIDEV=m/' .config
sed -i 's/CONFIG_LOCALVERSION=""/CONFIG_LOCALVERSION="-tegra"/' .config

make prepare
make modules_prepare
make M=drivers/spi/

cp drivers/spi/spidev.ko /lib/modules/$(uname -r)/kernel/drivers
depmod
#reboot 後でまとめて実行
```

## Device tree
メモ
```
# /boot/dtb/tegra186-quill-p3310-1000-c03-00-base.dtb をデコンパイルする
# tegra186-quill-p3310-1000-c03-00-base.dts ファイルが出来上がる
#spi0 = "/spi@3210000";
#spi1 = "/spi@c260000";
#spi2 = "/spi@3230000";
#spi3 = "/spi@3240000";
```

## DTC Tool [*D]
```
apt-get update
apt-get install device-tree-compiler
```

## SPI Configuration [*B]
```
cd /boot/dtb
# backup
cp tegra186-quill-p3310-1000-c03-00-base.dtb tegra186-quill-p3310-1000-c03-00-base.dtb.bak
# decompile
dtc -I dtb -O dts -o tegra186-quill-p3310-1000-c03-00-base.dts tegra186-quill-p3310-1000-c03-00-base.dtb

# edit tegra186-quill-p3310-1000-c03-00-base.dts
# spi@3240000 に書き加える
# The SPI pin group in the J21 looks like map to TX2 SPI4, You may need enable the spidev node to spi@3240000 [*B]
vi tegra186-quill-p3310-1000-c03-00-base.dts
        spi@3240000 {
                compatible = "nvidia,tegra186-spi";
                reg = <0x0 0x3240000 0x0 0x10000>;
                interrupts = <0x0 0x27 0x4>;
                nvidia,dma-request-selector = <0x19 0x12>;
                #address-cells = <0x1>;
                #size-cells = <0x0>;
                #stream-id-cells = <0x1>;
                dmas = <0x19 0x12 0x19 0x12>;
                dma-names = "rx", "tx";
                nvidia,clk-parents = "pll_p", "clk_m";
                clocks = <0xd 0x4a 0xd 0x10d 0xd 0x261>;
                clock-names = "spi", "pll_p", "clk_m";
                resets = <0xd 0x2b>;
                reset-names = "spi";
                status = "okay";
                linux,phandle = <0x80>;
                phandle = <0x80>;
                # add this
                spidev@0 {
                            compatible = "spidev";
                            reg = <0>;
                            spi-max-frequency=<25000000>;
                };
                spidev@1 {
                            compatible = "spidev";
                            reg = <1>;
                            spi-max-frequency=<25000000>;
                };
        };
# compile
dtc -I dts -O dtb -o tegra186-quill-p3310-1000-c03-00-base.dtb tegra186-quill-p3310-1000-c03-00-base.dts
```
## dtb into /dev/mmcblk0p15 [*A]
```
dd if=/boot/dtb/tegra186-quill-p3310-1000-c03-00-base.dtb of=/dev/mmcblk0p15
```
## reboot
```
reboot
```
## check spidev
```
lsmod
```
>Module                  Size  Used by  
>fuse                   89008  2  
>bcmdhd               7625819  0  
>spidev                 10966  0 ←これが出現した  
>pci_tegra              74691  0  
>bluedroid_pm           13564  0  

```
ls /dev/spi*
```
>/dev/spidev3.0  /dev/spidev3.1  

## 参考
  * [\*C] https://elinux.org/Jetson/TX1_SPI#Installing_SPIdev_Kernel_Module
  * [\*D] https://elinux.org/Jetson/TX1_SPI#Installing_DTC_Tool
  * [\*B] https://devtalk.nvidia.com/default/topic/1008929/jetson-tx2/enabling-spi-and-spidev-on-the-tx2/
  * [\*A] https://devtalk.nvidia.com/default/topic/1023007/how-to-use-uart0-as-normal-uart-port-on-r28-1-/?offset=12

