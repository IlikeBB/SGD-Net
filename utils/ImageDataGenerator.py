from utils.augmented import generator
def LN_generator(x_trains, y_trains, x_vals, y_vals, batch_size):
    # print('build generator')
    image_aug = generator.customImageDataGenerator(
                rotation_range = 10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                # shear_range=0.1,
                zoom_range=0.1,
                fill_mode='constant',
                horizontal_flip=True
                )
    image_aug_valid = generator.customImageDataGenerator(
                )
    train_generator = image_aug.flow(x_trains, y_trains, batch_size= batch_size, seed=123)
    valid_generator = image_aug_valid.flow(x_vals, y_vals, batch_size= batch_size, seed=123)
    return train_generator, valid_generator

def AP_generator(x_trains, y_trains, x_vals, y_vals, batch_size):
    image_aug = generator.customImageDataGenerator(
                rotation_range = 5,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    #             shear_range=0.2,
    #             zoom_range=0.2,
                fill_mode='constant',
                horizontal_flip=True
                )
    image_aug_valid = generator.customImageDataGenerator(
                )
    train_generator = image_aug.flow(x_trains, y_trains, batch_size= batch_size, seed=123)
    valid_generator = image_aug_valid.flow(x_vals, y_vals, batch_size= batch_size, seed=123)
    return train_generator, valid_generator