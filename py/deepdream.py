from keras.applications import inception_v3
from tensorflow.keras import backend as K
import numpy as np
import scipy
from keras.preprocessing import image
import imageio
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights = 'imagenet', include_top = False)

layer_contributions = {
    'mixed2': 0.1,
    'mixed3': 0.1,
    'mixed4': 0.1,
    'mixed5': 0.1,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    loss = loss + (coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)

dream = model.input
grads = K.gradients(loss, dream)[0]
grads = grads/(K.maximum(K.mean(K.abs(grads)), 1e-7))

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    """returns the loss and gradient values"""
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def gradient_ascent(x, iterations, step, max_loss=None):
    """Implements gradient access for a specified number of iterations"""
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


step = 0.1 #Step size for gradient ascent
num_octave = 2 #number of octaves to be run
octave_scale = 2 #this is the scale for each ensuing octive will be 1.4 times large than the previous
iterations = 20 #number of gradient ascent operations we execute 
max_loss = 20.0 #our early stoping metric, if loss is > max_loss we break the gradient ascent loop
base_image_path_nm = 'galaxy-11098.jpg'
base_image_path = '/home/diablo/Documents/dream/'+base_image_path_nm

# Load our image 
img = preprocess_image(base_image_path)

# Initialize a list of tuples for our different images sizes/scales 
original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# Reverse list of shapes, so that they are in increasing order
successive_shapes = successive_shapes[::-1]

# Resize the Numpy array of the image to our smallest scale
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)

    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='/home/diablo/Documents/dreamf/dreamscale_' + str(shape) + '.png')
    
save_img(img, fname='/home/diablo/Documents/dreamf/'+base_image_path_nm)
print("DeepDreaming Complete")