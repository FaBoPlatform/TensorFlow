# TensorFlowDocs

# 本ドキュメントについて

機械学習のプログラム学習用のドキュメントになっています。

# 対象レベル

プログラムを過去に書いたことがある

# 使用フレームワーク

TensorFlow

![](http://g.gravizo.com/g?
  digraph G {
    CoreData [shape=box]
    FLMAssetFetchService [shape=box]
    AppDelegate -> FLMAssetFetchService [label="1.start"]
    FLMAssetFetchService -> FLMAssetFetchOperation [label="2.Kick"]
    FLMAssetFetchOperation -> FLMAssetFetchService [label="3.NSNotification(FetchedAsset)"]
    FLMAssetFetchOperation -> FLMFetchedAssetTransformer [label="ALAsset or PHAsset"]
    FLMFetchedAssetTransformer -> FLMAssetFetchOperation [label=FLMFetchedAsset]
    FLMAssetFetchService -> "FLMAssetStoreOperation" [label="4.FetchedAsset"]
    "FLMAssetStoreOperation" -> CoreData [label="5.Commit FLMPhoto, FLMAlbum"]
    "FLMAssetStoreOperation" -> App [label="6.Notify via NSNotification"]
  }
)
