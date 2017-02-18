# プロトコルバッファ Part1

プロトコルバッファ(protocol buffer、protobuf)は、Google社によって開発されたデータの交換や保存に用いられるシリアライズフォーマットである。プロトコルバッファ形式で記述されたデータは、XMLやJSONと同じように複数のプログラミング言語で共有することが可能で、拡張子`.proto`のファイルとして保存される。

[公式のドキュメント](https://developers.google.com/protocol-buffers/docs/overview)によればプロトコルバッファはXMLと比較して以下の点で優れているとしている。

* よりシンプル
* 3倍から10倍サイズが小さい
* 20倍から100倍速い
* 曖昧性がより低い

## プロトコルバッファの簡単な例

プロトコルバッファ・XML・JSONの比較を以下に載せる。

* 型
* ユーザ定義型
* required、optional修飾子

Protocol buffer：

```proto
person {
  required name: strings "Taro Tanaka"
  optional age: int32 18
  required email: strings "taro@email.com"
}
```

XML：

```xml
<person>
	<name>Taro Tanaka</name>
	<age>18</age>
	<email>taro@email.com</email>
</person>
```

JSON：

```json
{
	"person": {
		"name": "Taro Tanaka",
		"age": 18,
		"email": "taro@email.com"
	}
}
```

## 参考

* [Protocol Buffers - Google Developers](https://developers.google.com/protocol-buffers/)
* [google/protobuf](https://github.com/google/protobuf)
* [5 Reasons to Use Protocol Buffers Instead of JSON For Your Next Service](http://blog.codeclimate.com/blog/2014/06/05/choose-protocol-buffers/)
* [Protocol Buffers - Wikipedia](https://ja.wikipedia.org/wiki/Protocol_Buffers)
* [Protocol Buffers 入門](http://www.slideshare.net/yuichi110/protocol-buffers-61413028)