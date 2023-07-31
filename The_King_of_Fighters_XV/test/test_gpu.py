import torch 

if __name__ == "__main__":
    print('TORCH VISION: {}'.format(torch.__version__))

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print('CUDA Device {} Name: {}'.format(i, torch.cuda.get_device_name(i)))
            print('CUDA Device {} Device_Capability: {}'.format(i, torch.cuda.get_device_capability(i)))
            # print all memory in GBs
            print('CUDA Device {} Memory: {} GB'.format(i, torch.cuda.get_device_properties(i).total_memory/1e9))
    else:
        print("No CUDA devices found")