from tensorflow.keras import layers, Model, Input


# ⭐Network embedding process
def _dense_emb(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        dense2 = layers.Dense(units=units_2, activation='relu')(dense1)
        bn = layers.BatchNormalization()(dense2)
        return bn

    return f


# 🍒Subcellular localization process
def _dense_bn_sub(**params):
    units_1 = params["units_1"]
    units_2 = params["units_2"]
    units_3 = params["units_3"]

    def f(ip):
        dense1 = layers.Dense(units=units_1, activation='relu')(ip)
        dense2 = layers.Dense(units=units_2, activation='relu')(dense1)
        bn = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(units=units_3, activation='relu')(bn)
        return dense3

    return f


# 🍩Gene expression process
def _depth_sep(**params):
    filters = params['filters']
    kernel_size = params['kernel_size']
    activation = params['activation']
    units = params['units']
    pool_size = params['pool_size']
    depth_multiplier = params['depth_multiplier']

    def f(ip):
        # Default channel_last=True
        batch_size, timestep, replicate, channel_num = ip.shape
        c_list = []
        # 1D convolution on each channel
        for i in range(channel_num):
            channel_data = ip[:, :, :, i]
            print(channel_data.shape)
            op = layers.Conv1D(filters=depth_multiplier, kernel_size=kernel_size, activation=activation)(channel_data)
            op = layers.BatchNormalization()(op)
            op = layers.MaxPool1D(pool_size=pool_size)(op)
            c_list.append(op)
        c_layer = layers.Concatenate(axis=-1)(c_list)
        # pointwise process on the concatenated output
        conv11 = layers.Conv1D(filters=filters, kernel_size=1, activation=activation)(c_layer)
        bn = layers.BatchNormalization()(conv11)
        flatten = layers.GlobalMaxPool1D()(bn)
        dense = layers.Dense(units=units, activation=activation)(flatten)
        return dense

    return f


class SPEP(object):
    @staticmethod
    def build(input_shape_gse, input_shape_emb, input_shape_sub):
        # 🐱 embedding part
        input_emb = Input(shape=input_shape_emb, name='Embedding')
        output_emb = _dense_emb(units_1=16, units_2=16)(input_emb)

        # 🐱 subloc part 
        input_sub = Input(shape=input_shape_sub, name='Subloc')
        output_sub = _dense_bn_sub(units_1=64, units_2=64, units_3=16)(input_sub)

        # 🐱 gse part
        input_gse = Input(shape=input_shape_gse, name='GeneExpression')
        output_gse = _depth_sep(filters=64, kernel_size=2, activation='relu',
                                units=16, pool_size=2, depth_multiplier=64)(input_gse)

        concat = layers.Concatenate(axis=-1)([output_emb, output_sub, output_gse])
        output = layers.Dense(units=1, activation='sigmoid')(concat)

        model = Model(inputs={'Embedding': input_emb, 'Subloc': input_sub,
                              'GeneExpression': input_gse}, outputs=output)
        return model


model = SPEP.build(input_shape_gse=(8, 3, 2), input_shape_sub=(1024,),
                   input_shape_emb=(64,))
model.summary()