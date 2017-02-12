

## Hexagonを有効にしたBuild

```shell
$ ./build_all_android.sh -X
```

Hexagon用のライブラリ

|ライブラリ名|DL先|
|:--|:--|
| libhexagon_controller.so | https://storage.googleapis.com/download.tensorflow.org/deps/hexagon/libhexagon_controller.so |
| libhexagon_nn_skel.so | https://storage.googleapis.com/download.tensorflow.org/deps/hexagon/libhexagon_nn_skel.so |

```
download_and_push() {
    URL="$1"
    LOCAL_DEST="$2"
    ANDROID_DEST="$3"
    curl -Ls "${URL}" -o "${LOCAL_DEST}"
    adb shell mkdir -p "${ANDROID_DEST}"
    adb push "${LOCAL_DEST}" "${ANDROID_DEST}"
}
```

で、adbで`libhexagon_controller.so`と`libhexagon_nn_skel.so`をadb pushする処理が入っているので、デバイスを接続しておく。

現在のデバイスでは、libhexagon_nn_skel.soを実機に転送する際に、

```shell
$ adb shell mkdir /vendor/lib/rfsa/adsp
```

で

```shell
mkdir: '/vendor/lib/rfsa': Read-only file system
```

となり、先に進めない。
