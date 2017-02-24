# RaspberryPi3 (OS:Raspbian) ネットワーク接続
クローズドネットワークで、RaspberryPi上で起動したjupyterにPCからアクセスしたい<br />
ネットワーク機器はRaspberryPiの起動で電源ONとなるUSB給電タイプを選択<br />
RaspberryPi3はWiFiモジュール内臓なのでWiFiルータも選択可能<br />
ちびファイはすべてハードウェア仕様が異なるので注意<br />


---
### ルータ <- WiFi -> RaspberryPi<br />ルータ <- WiFi -> PC
---
対応：`ちびファイ (MZK-RP150NA)` `ちびファイ2 (MZK-UE150N)` `ちびファイ2ac (MZK-UE450AC)`
>ちびファイ3 (MZK-DP150N,MZK-DP150N/R)<br />
>MZK-DP150N/Rは箱のラベルで本体型番はMZK-DP150Nのみ<br />
>ちびファイ3はUSB給電出来ないため対象外<br />

* #####  RaspberryPi側
  * 固定IP設定(wlan0)
  * WiFi接続先設定
  * 設定反映
* ##### PC側
  * WiFi接続
___
固定IP設定(wlan0)
```
sudo vi /etc/dhcpcd.conf
interface wlan0
static routers=192.168.111.1
static ip_address=192.168.111.100/24
```
・interfaceはwlan0。(USB受信機とかで)無線受信機を2個も3個も積んでいるRaspberyPiなら受信機に割り振られてそうな番号。(see iwconfig,iwlist,lsusb,lsmod)<br />
・routersはルータのIPアドレス。ちびファイの工場出荷設定は192.168.111.1<br />
・ip_addressはRaspberryPiで使いたいIPアドレス192.168.111.100と、/24でIPアドレスに関連するルーティングプレフィックス192.168.100.0、またはそれに相当するサブネットマスク255.255.255.0を指定し、自分のIPアドレスは192.168.111.100、所属しているサブネット(同一ネットワークのアドレス範囲)は192.168.111.0から192.168.111.255の範囲であることを指定<br />
・192.168.111.0はネットワークアドレス、192.168.111.255はブロードキャストアドレス、192.168.111.1はちびファイに割り振られているので、利用可能なIPアドレスはそれ以外の192.168.111.2から192.168.111.254の範囲になる<br />
・今回はクローズドネットワークでの利用なのでDNS設定は無くても問題ない<br />
___
WiFi接続先設定
```ruby:qiita.rb
sudo vi /etc/wpa_supplicant/wpa_supplicant.conf

country=GB
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
    ssid="planex-xxxxxx"
    proto=WPA2
    key_mgmt=WPA-PSK
    psk="xxxxxxxxxx"
}
```
・ssidはちびファイのssid(gameじゃない方)<br />
・pskは鍵マークの英数字。暗号化したパスワードも可能(see [wpa_passphrase](https://linux.die.net/man/8/wpa_passphrase))
___
WiFiアクティベート
```
# wifiアクティベート
ip link set wlan0 up
ip link show dev wlan0
```
設定反映
```
# wifi停止
sudo ipdown wlan0
# wifi起動
sudo ipup wlan0
# IPアドレス確認
ifconfig -a
```


---
### ルータ <- LAN -> RaspberryPi<br />ルータ <- WiFi -> PC
---
対応：`ちびファイ (MZK-RP150NA)`
* #####  RaspberryPi側
  * ちびファイ LAN側ポートとRaspberryPi LANポートを接続
  * 固定IP設定(eth0)
  * 設定反映
* ##### PC側
  * WiFi接続


---
### ルータ <- USB -> RaspberryPi<br />ルータ <- WiFi -> PC
---
対応：`ちびファイ2 (MZK-UE150N)`
* #####  RaspberryPi側
  * ちびファイ2をRaspberryPiにUSB接続する
  * 固定IP設定(eth1)
  * 設定反映
* ##### PC側
  * WiFi接続




