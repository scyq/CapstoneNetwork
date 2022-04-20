# -*- coding: utf-8 -*-

import torch


class Config:
    '''
    Chatbot模型参数
    '''
    # corpus_data_path = 'corpus.pth'  #已处理的对话数据
    corpus_data_path = "corpus_new.pth"
    use_QA_first = True  #是否载入知识库
    max_input_length = 50  #输入的最大句子长度
    max_generate_length = 20  #生成的最大句子长度
    prefix = 'checkpoints/chatbot'  #模型断点路径前缀
    model_ckpt = 'checkpoints/chatbot_0419_1533'  #加载模型路径
    # model_ckpt = None  # 如果从零开始训练, 则不从任何checkpoints继续
    '''
    训练超参数
    '''
    # batch_size = 2048
    batch_size = 1024
    shuffle = True  #dataloader是否打乱数据
    num_workers = 0  #dataloader多进程提取数据
    bidirectional = True  #Encoder-RNN是否双向
    hidden_size = 256
    embedding_dim = 256
    method = 'dot'  #attention method
    dropout = 0  #是否使用dropout
    clip = 50.0  #梯度裁剪阈值
    num_layers = 2  #Encoder-RNN层数
    learning_rate = 1e-3
    teacher_forcing_ratio = 1.0  #teacher_forcing比例
    decoder_learning_ratio = 5.0
    '''
    训练周期信息
    '''
    epoch = 300
    save_every = 1  #每隔save_every个Epoch打印一次
    '''
    GPU
    '''
    use_gpu = torch.cuda.is_available()  #是否使用gpu
    device = torch.device("cuda" if use_gpu else "cpu")  #device


if __name__ == "__main__":
    print(torch.version.__version__)
    print(torch.cuda.is_available())