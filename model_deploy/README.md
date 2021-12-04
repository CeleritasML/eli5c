# How to Deploy ELI5C LFQA Model

In this folder, the model is wrapped in a Flask backend. In this passage, I will introduce how to deploy this model.

## 1 Environments

Install the `requirements.txt` in the root folder and also install `Flask` library.

## 2 Models

To make things easier, the pre-trained BERT retriever and BART generator had been uploaded to HuggingFace (BERT: [jsgao/bert-eli5c-retriever](https://huggingface.co/jsgao/bert-eli5c-retriever), BART: [jsgao/bart-eli5c](https://huggingface.co/jsgao/bart-eli5c)). The program will automatically download and load them.

The pre-trained additional linear projection layers in the retriever are also dumped by torch and contained in `models/bert_eli5c_projection.pt`. Again, the program will automatically load pre-trained parameters for these layers.

## 3 Wikipedia Index

The pre-computed wikipedia indexes need to be downloaded from Google Drive [wiki40b_index.bin](https://drive.google.com/file/d/1-ik5uQkyYjbgytgFrKLTbK7Idcwo49Cl/view?usp=sharing) (file size: 8.50 GB) and put it into the `models/` folder.

**The binary file is a trained faiss IndexIVFFlat object, the object may not be able to work when using a different faiss version.** If that happens, try to download the embedding vectors file from Google Drive [wiki40b.dat](https://drive.google.com/file/d/1ywlO_3x3RcYwO6kdwWHICyxXjZM6s9x2/view?usp=sharing) and put it into `/model_code/embeds/` folder and re-run `/model_code/retriever_bert_doc_query.py` to get a new `wiki40b_index.bin` file for your faiss version.

## 4 Deploy Model

Run `python main.py` at this folder and the model will be deployed using Flask. **ATTENTION: During loading, the program can take more than 25 GB RAM at peak. After deployed, the program takes up to 12 GB RAM and 5 GB GPU memory** 

## 5 Send Requests

After deployed model, you can send `POST` requests with `{"question": "<some question>"}` to the server. 

Also, I provide a client html `model-ajax-client.html` in this folder that using AJAX to send this request to the server. Change the server address at `line 103` if needed.