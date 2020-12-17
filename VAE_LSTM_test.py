from VAE.VAE_LSTM import VAE
from Dataloader.smile_sequence_loader import Sequence_DataLoader
import numpy as np
#from Dataloader.smile_sequence_loader import Sequence_DataLoader
import matplotlib.pyplot as plt

filename="Dataset\hiv_inhibitors.smi"
data_loader=Sequence_DataLoader(data_filename=filename, data_type='train')

vae_models=VAE()
print(vae_models.vae.summary())
vae_models.vae.compile(optimizer='adam', loss='binary_crossentropy')

history=vae_models.vae.fit(
    data_loader,
    epochs=5,
    batch_size=128,
    verbose=1,
    shuffle=True,
)

for key in history.history.keys():
    plt.plot(history.history[key])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

vae_models.decoder.save("Weights/VAE_LSTM_Decoder.h5")

test_input=[]
for i in range(data_loader.max_len):
    test_input.append(np.random.uniform(low=-10,high=10,size=(vae_models.latent_space_dim)))
test_input=np.array(test_input)
tmp=vae_models.decoder.predict(test_input.reshape((1,data_loader.max_len,vae_models.latent_space_dim)))
print(tmp)
