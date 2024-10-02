# %%
from src import inference_confidence
from model.QA_bert_dropout_ver import Contextual_BERT
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from data_process import dataset
import torch
from src import config

# %%
def average_non_zero(lst):
    """
    Calculate the average of non-zero values in the list.
    Returns 0 if the list contains no non-zero values.
    """
    non_zero_values = list(filter(lambda x: x != 0, lst))
    return sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0

def load_model(path='"clw8998/Product-Name-NER-model', device=torch.device('cpu')):
    """
    Load the model and tokenizer from the specified path and move the model to the specified device.
    """
    model = Contextual_BERT.from_pretrained(path).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(path)
    return model, tokenizer

def inference(model, tokenizer, inference_data, all_attribute, batch_size=32):
    """
    Perform inference on the input data using the specified model and tokenizer.
    Returns the inference results.
    """
    if not inference_data:
        return None

    # Preprocess the inference data
    processed_data = [{'context': context, 'question': attr, 'answer': []}
                      for context in inference_data
                      for attr in all_attribute]

    # Create dataset and data loader for inference
    inference_dataset = dataset.BERTDataset_preprocess(processed_data, [], tokenizer)
    inference_loader = DataLoader(
        dataset=inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.BERTDataset_preprocess.collate_fn
    )

    # Run inference and return the result
    return inference_confidence.inference(inference_loader, model, tokenizer, all_attribute)

def process_result(result):
    """
    Process the inference result to extract attributes and their corresponding confidence scores.
    Returns a dictionary with the processed results.
    """
    result_dict = {}

    # Iterate through the result data
    for attributes_data, indexes_data, confidences_data in zip(result[0], result[1], result[2]):
        title, attributes = attributes_data
        _, indexes = indexes_data
        _, confidences = confidences_data
        att_dict = {}

        # Process each attribute
        for att_key, attribute_values in attributes.items():
            att_confs = []
            spans = indexes[att_key]
            confidence = confidences[att_key]

            # Calculate the average confidence for each span
            for span in spans:
                att_confs.append(round(average_non_zero(confidence[span[0]:span[1] + 1]), 4))

            # Match attribute values with their corresponding confidence scores
            att_result = [[attribute, att_confs[j]] for j, attribute in enumerate(attribute_values)]
            att_dict[att_key] = att_result

        result_dict[title] = att_dict

    return result_dict

if __name__ == '__main__':
    # %%
    # put attribute here!
    all_attribute = ['品牌', '名稱', '產品', '產品序號', '顏色', '材質', '對象與族群', '適用物體、事件與場所', 
                        '特殊主題', '形狀', '圖案', '尺寸', '重量', '容量', '包裝組合', '功能與規格']

    # put infernce data here!
    inference_data = ['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊', '【Jabra】Elite 4 ANC真無線降噪藍牙耳機 (藍牙5.2雙設備連接)']

    # set device
    config.string_device =  'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = torch.device(config.string_device)

    # load model
    model, tokenizer = load_model("clw8998/Product-Name-NER-model", device=config.device)

    # inference
    result = inference(model, tokenizer, inference_data, all_attribute, batch_size=32)

    # process result
    result_dict = process_result(result)

    # %%
    # use inference data to get result (Should be lower case)
    print(inference_data[0])
    result_dict[inference_data[0].lower()]

    # %%
    # use inference data to get result (Should be lower case)
    print(inference_data[1])
    result_dict[inference_data[1].lower()]

    # %%
    print('【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊')
    result_dict['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊'.lower()]

    # %%
    print('【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊')
    result_dict['【A‵bella浪漫晶飾】方形密碼-深海藍水晶手鍊'.lower()]['品牌']


