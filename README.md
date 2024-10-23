This is an example of the homework only for who notice this branch :) To run this, just follow the instructions below:
1. Move your Mxxxxxxxx_xxx_assigned_queries.csv to the root directory.
2. Run NER-relevancy-example.ipynb

You will see a result csv named Mxxxxxxxx-NER-relevancy.csv

---

# Product-Name-NER-for-EE5327701

[English](./README_en.md)

此專案包含一個**繁體中文**命名實體識別 (NER) 模型，可用於提取**商品名稱**中的**各種屬性**，例如品牌、名稱、顏色等。

---

## 目錄

1. [安裝](#1-安裝)
2. [推論範例](#2-推論範例)
3. [NER 自動相關性評估作業](#3-ner-自動相關性評估作業) **(重點)**
4. [屬性標註規則及其準確度](#4-屬性標註規則及其準確度)

## 1. 安裝

### 1.1 PyTorch 安裝

建議從 [PyTorch 官方網站](https://pytorch.org/get-started/locally/) 進行安裝。

### 1.2 安裝依賴套件

使用以下命令安裝專案所需的依賴套件：

```bash
pip install -r requirements.txt
```

## 2. 推論範例

推論範例寫在 `inference_example.ipynb` 中，主要可調整的內容包括輸入的商品名稱與想要提取的 NER 屬性。以下為基本推論程式範例：

```python
# put attribute here!
all_attribute = ['品牌', '名稱', '產品', '產品序號', '顏色', '材質', '對象與族群', '適用物體、事件與場所', 
                     '特殊主題', '形狀', '圖案', '尺寸', '重量', '容量', '包裝組合', '功能與規格']

# put infernce data here!
inference_data = ['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊', '【Jabra】Elite 4 ANC真無線降噪藍牙耳機 (藍牙5.2雙設備連接)']

# set device
config.string_device =  'cuda' if torch.cuda.is_available() else 'cpu'
config.device = torch.device(config.string_device)

# load model
model, tokenizer = inference_api.load_model("clw8998/Product-Name-NER-model", device=config.device)

ner_tags = inference_api.get_ner_tags(model, tokenizer, inference_data, all_attribute)
```

### 2.1 部分推論結果

推論結果將包含結構化的屬性資料及其對應的信心分數，範例如下：

```json
{
  "【a‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊": {
    "品牌": [["a‵bella", 0.9987], ["浪漫晶飾", 0.9861]],
    "名稱": [["密碼", 0.6318]],
    "顏色": [["深海藍", 0.9486]],
    "材質": [["水晶", 0.9143]],
    ...
  },
  "【Jabra】Elite 4 ANC真無線降噪藍牙耳機 (藍牙5.2雙設備連接)": {
    "品牌": [["Jabra", 0.9989]],
    "名稱": [["Elite 4", 0.7890]],
    "功能與規格": [["ANC", 0.9523], ["藍牙5.2", 0.8991]],
    "產品": [["藍牙耳機", 0.9981]],
    ...
  }
}
```

## 3. NER 自動相關性評估作業 **(重點)**

在此作業中，學生將根據給定的商品名稱和屬性，利用 NER 模型來自動化評估查詢詞與商品名稱之間的相關性。`NER-relevancy-example.ipynb` 提供了一個範例來展示如何使用 `ner_relevancy()` 函數進行相關性計算。

### 3.1 `ner_relevancy()` 函數介紹

此函數用於比較查詢詞的 NER 標籤與從不同模型（如 TF-IDF 和語義搜索模型）生成的標籤，並計算相關性分數。相關性分數是基於查詢標籤與模型標籤之間的交集大小來評估的，透過設定的 margin 閾值來決定分數等級。

### 3.2 主要步驟

1. **提取 NER 標籤**：
   - `query_tags_dict`：查詢詞的 NER 標籤結果。
   - `tfidf_tags_dict`：TF-IDF 模型的 NER 標籤結果。
   - `semantic_tags_dict`：語義模型的 NER 標籤結果。
   
   這些標籤結果存儲在字典中，其中每個屬性對應的值是標籤的名稱和信心分數。

2. **生成標籤集合**：
   - 從每個標籤字典中提取標籤名稱，形成集合：
     - `query_tags_pool`：查詢詞的標籤集合。
     - `tfidf_tags_pool`：TF-IDF 模型的標籤集合。
     - `semantic_tags_pool`：語義模型的標籤集合。

3. **計算查詢標籤與 TF-IDF 標籤的相關性**：
   - 透過比較 `query_tags_pool` 與 `tfidf_tags_pool` 的交集大小來決定相關性：
     - 若交集大小大於或等於給定標籤數量和 margin 閾值的乘積，則相關性分數為 2（高相關）。
     - 若交集大小大於或等於 margin 閾值的一半，則相關性分數為 1（中等相關）。
     - 否則，相關性分數為 0（無相關）。

4. **計算查詢標籤與語義模型標籤的相關性**：
   - 使用與上一步相同的比較方法來計算 `query_tags_pool` 與 `semantic_tags_pool` 的相關性分數。

### 3.3 自動化相關性計算的意義

此函數可以幫助學生自動化比較查詢詞與商品名稱的 NER 標籤重疊程度，並進一步計算相關性分數。同學可以根據這些分數調整 margin 閾值或改進評估算法，以優化 NER 模型的自動化評估流程。

## 4. 屬性標註規則及其準確度

以下是 NER 模型的屬性標註規則及對應的 F1-score：

| 編號 | 標籤屬性            | 說明                                                                                          | F1-score |
|-----|-------------------|---------------------------------------------------------------------------------------------|-------|
| 1   | **品牌**            | 商品品牌名稱，如 華碩、LG                                                                     | 0.8770 |
| 2   | **名稱**            | 商品中的「產品系列」或特別創造出的商品名稱，不包含特殊主題或產品類型，如 iPhone 12、ROG 3060Ti | 0.5941 |
| 3   | **產品**            | 實際產品名稱（類型），如 電腦、滑鼠、鍵盤、玩具、餅乾                                         | 0.8001 |
| 4   | **產品序號**        | 商品序號，該產品的唯一數字字母組合序號，不含系列名                                             | 0.8443 |
| 5   | **顏色**            | 顏色資訊，包含如 花朵紅、藍色系、晶亮                                                            | 0.8898 |
| 6   | **材質**            | 產品的製造材料，如 木製、PVC 材質、304 不銹鋼                                                   | 0.7810 |
| 7   | **對象與族群**        | 商品對象或族群，如 新生兒、寵物用、高齡族群                                                    | 0.8755 |
| 8   | **適用物體、事件與場所** | 適用的物品、事件或場所，如 手部用、騎車用、廚房用                                             | 0.7614 |
| 9   | **特殊主題**        | 商品附有的特殊主題，如 航海王、J.K. Rowling                                                     | 0.4979 |
| 10  | **形狀**            | 商品的形狀或外觀詞彙，如 圓形、鈕扣形、無袖、窄邊框                                             | 0.7793 |
| 11  | **圖案**            | 產品上的圖案，如 圓形、螺旋形圖案                                                             | 0.5731 |
| 12  | **尺寸**            | 商品大小，如 120x80x10cm、XL、ATX                                                              | 0.8420 |
| 13  | **重量**            | 商品重量，如 10g、極輕                                                                       | 0.9105 |
| 14  | **容量**            | 商品容量，如 128GB（電腦）、大容量                                                             | 0.9220 |
| 15  | **包裝組合**        | 產品的包裝或組合方式，如 10入、鍵盤滑鼠組合、送電池                                             | 0.7478 |
| 16  | **功能與規格**        | 產品的用途或特殊規格，如 USB3.0、防水                                                          | 0.7960 |
| 17  | **Macro F1-score**   |                                                                                             | 0.7807 |