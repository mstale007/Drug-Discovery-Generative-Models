from smile_sequence_loader import Sequence_DataLoader

filename="hiv_inhibitors.smi"
data_loader=Sequence_DataLoader(data_filename=filename, data_type='train')

print('First Element:')
print(data_loader.__getitem__(0))