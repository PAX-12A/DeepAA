# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import numpy as np
# import pandas as pd
# import os, math, time
# from sklearn.model_selection import train_test_split


# # =========================
# # 1️⃣ 数据加载部分
# # =========================
# class CharDataset(Dataset):
#     def __init__(self, df, img_dir, input_size, tail='line_resize', slide=1):
#         self.df = df.reset_index(drop=True)
#         self.img_dir = img_dir
#         self.input_size = input_size
#         self.tail = tail
#         self.slide = slide

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         x, y = row['x'] + 30, row['y'] + 30
#         img_path = os.path.join(self.img_dir, row['file_name'] + self.tail + '.png')

#         img = np.array(Image.open(img_path).convert('L'))
#         h, w = img.shape
#         mergin = (self.input_size - 18) // 2 + 30

#         # add margin
#         img_new = np.ones((h + 2*mergin, w + 2*mergin), dtype=np.uint8) * 255
#         img_new[mergin:-mergin, mergin:-mergin] = img

#         # random slide
#         x += np.random.randint(-self.slide, self.slide + 1)
#         y += np.random.randint(-self.slide, self.slide + 1)

#         patch = img_new[y:y+self.input_size, x:x+self.input_size]
#         patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / 255.0
#         label = torch.tensor(row['label'], dtype=torch.long)
#         return patch, label


# # =========================
# # 2️⃣ 轻量CNN模型
# # =========================
# class LightCNN(nn.Module):
#     def __init__(self, num_classes=1000):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# # =========================
# # 3️⃣ 训练流程
# # =========================
# def train():
#     df_path = "data/data_500.csv"
#     char_list_path = "data/char_list_500.csv"
#     img_dir = "data/image_500/"
#     input_size = 32
#     batch_size = 128
#     num_epoch = 3
#     lr = 0.001

#     df = pd.read_csv(df_path, encoding='cp932')
#     char_list = pd.read_csv(char_list_path, encoding='cp932')
#     num_label = char_list[char_list['frequency'] >= 10].shape[0]
#     df = df[df['label'] < num_label]

#     df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

#     train_ds = CharDataset(df_train, img_dir, input_size)
#     val_ds = CharDataset(df_val, img_dir, input_size, slide=0)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = LightCNN(num_classes=num_label).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#     for epoch in range(num_epoch):
#         model.train()
#         total_loss, total_acc = 0, 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             total_acc += (outputs.argmax(1) == labels).float().sum().item()

#         print(f"[Epoch {epoch+1}/{num_epoch}] Train Loss: {total_loss/len(train_loader):.4f}  Acc: {total_acc/len(train_ds):.4f}")

#         # Validation
#         model.eval()
#         val_loss, val_acc = 0, 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 val_loss += criterion(outputs, labels).item()
#                 val_acc += (outputs.argmax(1) == labels).float().sum().item()
#         print(f"           Val Loss: {val_loss/len(val_loader):.4f}  Acc: {val_acc/len(val_ds):.4f}")

#     torch.save(model.state_dict(), "lightcnn.pth")
#     print("✅ Training finished and model saved as lightcnn.pth")


# if __name__ == "__main__":
#     train()


from keras.models import Model
from keras.layers import Dense, Activation, Reshape, Dropout, Embedding, Input, BatchNormalization
from keras.layers import Concatenate, Multiply, Conv2D, MaxPooling2D, Add, Flatten, GaussianNoise
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, \
    EarlyStopping, CSVLogger, ReduceLROnPlateau

import time
import numpy as np

np.random.seed(42)
import pandas as pd
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from multiprocessing import Pool

##
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def CBRD(inputs, filters=64, kernel_size=(3,3), droprate=0.5):
    x = Conv2D(filters, kernel_size, padding='same',
               kernel_initializer='random_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dropout(droprate)(x)
    return x


def DBRD(inputs, units=4096, droprate=0.5):
    x = Dense(units)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(droprate)(x)
    return x

def CNN(input_shape=None, classes=1000):
    inputs = Input(shape=input_shape)

    # Block 1（减少通道数量）
    x = GaussianNoise(0.1)(inputs)
    x = CBRD(x, 16)   # 原64
    x = MaxPooling2D()(x)

    # Block 2
    x = CBRD(x, 32)   # 原128
    x = MaxPooling2D()(x)

    # Block 3（删去一层 + 降通道）
    x = CBRD(x, 64)   # 原256
    x = MaxPooling2D()(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = DBRD(x, 256, droprate=0.3)  # 原4096
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def add_mergin(img, mergin):
    if mergin!=0:
        img_new = np.ones([img.shape[0] + 2 * mergin, img.shape[1] + 2 * mergin], dtype=np.uint8) * 255
        img_new[mergin:-mergin, mergin:-mergin] = img
    else:
        img_new = img
    return img_new


def load_img(args):
    img_path, x, y, input_size, mergin, slide = args
    img = np.array(Image.open(img_path))
    if len(img.shape) == 3:
        img = img[:, :, 0]
    img = add_mergin(img, mergin)
    x += np.random.randint(-slide, slide+1)
    y += np.random.randint(-slide, slide+1)
    img = img[y:y + input_size, x:x + input_size]
    img = img.reshape([1, input_size, input_size, 1])
    # print(img_path, x, y, input_size, mergin )
    # print(input_size, img.shape)
    return img

def batch_generator(df, img_dir, input_size, batch_size, num_label, slide,
                    tail='line', shuffle=True):
    df = df.reset_index()
    batch_index = 0
    mergin = (input_size - 18) // 2 + 30
    n = df.shape[0]
    pool = Pool()
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        index_array_batch = index_array[current_index: current_index + current_batch_size]
        batch_img_path = df['file_name'][index_array_batch].apply(
            lambda x: img_dir + x + tail + '.png').as_matrix()
        # print(batch_img_path)
        batch_coord_x = (df['x'][index_array_batch] + 30).as_matrix()
        batch_coord_y = (df['y'][index_array_batch] + 30).as_matrix()
        # print(batch_img_path[0], batch_coord_x[0], batch_coord_y[0], mergin)
        batch_x = pool.map(load_img,
                           [(batch_img_path[i],
                             batch_coord_x[i],
                             batch_coord_y[i],
                             input_size,
                             mergin,
                             slide)
                           for i in range(current_batch_size)])
        # print(batch_x[0].shape)
        batch_x = np.concatenate(batch_x, axis=0)
        batch_x = batch_x.astype(np.float32) / 255
        # print(batch_x.shape)

        batch_y = df['label'][index_array[current_index: current_index + current_batch_size]].as_matrix()
        batch_y = np.eye(num_label)[batch_y]

        yield batch_x, batch_y


def train_generator(df, img_dir, input_size, batch_size, num_label, slide,
                    tail='line', shuffle=True):
    gen_line = batch_generator(df, img_dir, input_size,
                               batch_size // 2, num_label, slide, tail="line_resize")
    gen_orig = batch_generator(df, img_dir, input_size,
                               batch_size // 2, num_label, slide, tail="orig")
    while True:
        batch1 = next(gen_line)
        batch2 = next(gen_orig)
        batch_x = np.concatenate([batch1[0], batch2[0]])
        batch_y = np.concatenate([batch1[1], batch2[1]])
        yield batch_x, batch_y


def train():
    # parameter
    # num_epoch = 256
    num_epoch = 3
    # batch_size = 64
    batch_size = 128
    # input_shape = [64,64,1]
    input_shape = [32, 32, 1]
    learning_rate = 0.001
    df_path = "data/data_500.csv"
    char_list_path = "data/char_list_500.csv"
    img_dir = "data/image_500/"

    # load text
    df = pd.read_csv(df_path, encoding="cp932")
    char_list = pd.read_csv(char_list_path, encoding="cp932")
    num_label = char_list[char_list['frequency']>=10].shape[0]
    # print(num_label)
    df = df[df['label']<num_label]
    df = df.reset_index()
    input_size = input_shape[0]
    slide = 1
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)
    gen = train_generator(df_train, img_dir,
                          input_size, batch_size, num_label, slide)
    gen_val = batch_generator(df_val, img_dir, input_size,
                              batch_size, num_label, 0,
                              tail="line_resize", shuffle=False)

    # build model
    model = CNN(input_shape=input_shape, classes=num_label)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # train
    nb_train = df_train.shape[0]
    nb_val = df_val.shape[0]
    nb_step = math.ceil(nb_train / batch_size)
    nb_val_step = math.ceil(nb_val / batch_size)

    format = "%H%M"
    ts = time.strftime(format)
    save_path = "model/Eden/" + path.splitext(__file__)[0] + "_" + ts
    #save_path = "model/" + path.splitext(__file__)[0] + "_" + ts

    json_string = model.to_json()
    with open(save_path + '_model.json', "w") as f:
        f.write(json_string)

    csv_logger = CSVLogger(save_path + '_log.csv', append=True)
    check_path = save_path + '_e{epoch:02d}_vl{val_loss:.5f}.hdf5'
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='val_loss', save_best_only=True)
    lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=learning_rate * 0.00001)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=16,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min')
    Callbacks = [csv_logger,
                 save_checkpoint,
                 lerning_rate_schedular, early_stopping]
    model.fit_generator(gen,
                        steps_per_epoch=nb_step,
                        epochs=num_epoch,
                        validation_data=gen_val,
                        validation_steps=nb_val_step,
                        callbacks=Callbacks
                        )


if __name__ == "__main__":
    train()
