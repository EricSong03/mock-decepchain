import torch, transformers, peft, trl, vllm
print('torch       ', torch.__version__)
print('cuda avail  ', torch.cuda.is_available())
print('device      ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('transformers', transformers.__version__)
print('trl         ', trl.__version__)
print('peft        ', peft.__version__)
print('vllm        ', vllm.__version__)