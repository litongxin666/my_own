from att_train import Trainer

dataset_path = '/home/litongxin'
batch_size = 64
num_workers = 4
epochs = 200
save_path = './result'

trainer = Trainer(dataset_path,batch_size, num_workers, epochs,save_path)
trainer.train()

