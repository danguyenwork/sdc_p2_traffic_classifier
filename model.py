import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import image
import pickle
import os

def load_data():
	train_file = LIST_DIR + '/train.p'
	test_file = LIST_DIR + '/test.p'
	valid_file = LIST_DIR + '/valid.p'

	with open(train_file, 'rb') as f:
	    train = pickle.load(f)
	with open(valid_file, 'rb') as f:
	    valid = pickle.load(f)
	with open(test_file, 'rb') as f:
	    test = pickle.load(f)

	x_train, y_train = train['features'], train['labels']
	x_valid, y_valid = valid['features'], valid['labels']
	x_test, y_test = test['features'], test['labels']

	x_new = []
	y_new = []

	new_img_dir = 'img/test_img/'
	arr = sorted(os.listdir(new_img_dir))
	for name in arr:
		if '.DS_Store' not in name:
			x = plt.imread(new_img_dir+name)
			y = int(name[:2])
			x_new.append(x)
			y_new.append(y)

	x_new = np.array(x_new)
	y_new = np.array(y_new)

	x_train = (x_train - 128.) / 128
	x_valid = (x_valid - 128.) / 128
	x_test = (x_test - 128.) / 128
	x_new = (x_new - 128.) / 128


	return x_train, y_train, x_valid, y_valid, x_test, y_test, x_new, y_new

def print_dimensions(data):
	x_train, y_train, x_valid, y_valid, x_test, y_test, x_new, y_new = data
	n_train = x_train.shape[0]
	n_valid = x_valid.shape[0]
	n_test = x_test.shape[0]
	n_classes = np.unique(y_train).shape[0]
	image_shape = x_train.shape[1]

	print("Number of training examples =", n_train)
	print("Number of validation examples =", n_valid)
	print("Number of testing examples =", n_test)
	print("Image data shape =", image_shape)
	print("Number of classes =", n_classes)

def _get_random_index_in_class(df, i):
    x = np.random.choice(df[df['class'] == i].index)
    # print(x)
    return x

def perform_eda(data):
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_new, y_new = data
    plot_dir = 'img/plots'
    signs_dir = 'img/signs'

    # line chart graph of the count of each class in each set
    df_train = pd.DataFrame(columns=['class'], data = y_train)
    df_train['train_cnt'] = 1
    df = df_train.groupby(['class']).count()

    df_valid = pd.DataFrame(columns=['class'], data = y_valid)
    df_valid['valid_cnt'] = 1
    df = pd.concat([df, df_valid.groupby(['class']).count()],axis=1)

    df_test = pd.DataFrame(columns=['class'], data = y_test)
    df_test['test_cnt'] = 1
    df = pd.concat([df, df_test.groupby(['class']).count()],axis=1)

    ax = df.plot(kind='line', title ="Count of signs by type",figsize=(15,10),xticks=df.index, legend=True, fontsize=12, rot=90)
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    ax.get_figure().savefig(plot_dir + '/count_chart.png')

    # save sample images in each class
    for i in np.unique(y_train):
        train_example = x_train[_get_random_index_in_class(df_train, i)]
        valid_example = x_valid[_get_random_index_in_class(df_valid, i)]
        test_example = x_test[_get_random_index_in_class(df_test, i)]

        image.imsave(signs_dir + '/'+str(i)+'_train.png', train_example)
        image.imsave(signs_dir + '/'+str(i)+'_valid.png', valid_example)
        image.imsave(signs_dir + '/'+str(i)+'_test.png', test_example)

def covnet(x, keep_prob):
	mu = 0
	sigma = 0.1

    # Convolutional Layer 1
    # Input 32 x 32 x 3
    # Output 28 x 28 x 6

	shape_conv1 = [5,5,3,6]

	conv1_w = tf.Variable(tf.truncated_normal(shape=shape_conv1, mean = mu, stddev = sigma))
	conv1_b = tf.Variable(tf.zeros([6]))
	conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1],padding='VALID') + conv1_b

    # Activation
	conv1 = tf.nn.relu(conv1)

    # Pooling
    # Input 28 x 28 x 6
    # Output 14 x 14 x 6
	conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Convolutional Layer 2
    # Input 14 x 14 x 6
    # Output 10 x 10 x 16

	conv2_w = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean = mu, stddev = sigma))
	conv2_b = tf.Variable(tf.zeros([16]))
	conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1],padding='VALID') + conv2_b

    # Activation
	conv2 = tf.nn.relu(conv2)

    # Pooling
    # Input 10 x 10 x 16
    # Output 5 x 5 x 16

	conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides = [1,2,2,1], padding='VALID')

    # Flatten
    # Input 5 x 5 x 16
    # Output 400

	fc0 = flatten(conv2)

	# Input 400
    # Fully Connected Layer 1
    # Output 120

	fc1_w = tf.Variable(tf.truncated_normal(shape=[400,120], mean = mu, stddev = sigma))
	fc1_b = tf.Variable(tf.zeros([120]))
	fc1 = tf.matmul(fc0, fc1_w) + fc1_b

    # Activation
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, keep_prob)

    # Fully Connected Layer 2
    # Input 120
    # Output 84

	fc2_w = tf.Variable(tf.truncated_normal(shape=[120, 84], mean = mu, stddev = sigma))
	fc2_b = tf.Variable(tf.zeros([84]))
	fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    # Activation
	fc2 = tf.nn.relu(fc2)
	fc2 = tf.nn.dropout(fc2, keep_prob)

    # Fully Connected Layer 3
    # Input 84
    # Output 43
	fc3_w = tf.Variable(tf.truncated_normal(shape=[84,43], mean = mu, stddev = sigma))
	fc3_b = tf.Variable(tf.zeros([43]))
	fc3 = tf.matmul(fc2, fc3_w) + fc3_b

	return fc3

def evaluate(x_valid, y_valid,batch_size_val):
	num_examples = x_valid.shape[0]
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, batch_size_val):
		end = offset + batch_size_val
		batch_x, batch_y = x_valid[offset:end], y_valid[offset:end]
		# import ipdb; ipdb.set_trace()
		batch_accuracy_score = sess.run(accuracy_score, feed_dict={x:batch_x, y:batch_y, keep_prob: 1.0})
		total_accuracy += batch_accuracy_score * batch_x.shape[0]
	return(total_accuracy / num_examples)

def final_model_evaluate(batch_size_val):
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph('traffic.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('.'))
		test_accuracy = evaluate(x_test, y_test, batch_size_val)
		print("  Test Accuracy = {:.3f}".format(test_accuracy))
		print()

		new_img_dir = 'img/test_img/'
		arr = sorted(os.listdir(new_img_dir))
		predicted_logits = sess.run(logits, feed_dict={x: x_new, keep_prob: 1.0})

		for i in range(x_new.shape[0]):
			print("--")
			predicted_label = np.argmax(predicted_logits[i])
			print('File name: \t\t\t', arr[i])
			print('Actual label: \t\t\t', y_new[i], ' - ', d[y_new[i]])
			print('Predicted label: \t\t', predicted_label, ' - ', d[predicted_label])
			print('Top softmax: ')
			print(sess.run(tf.nn.top_k(tf.constant(predicted_logits[i]), k=5)))
			print('Overall prob:')
			# np.set_printoptions(precision=2, suppress=True)

			softmax = np.exp(predicted_logits[i]) / np.exp(predicted_logits[i]).sum()

			softmax.sort()

			print(softmax[-5:])
# ========

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_DEPTH = 3
EPOCHS = 30
N_CLASSES = 43
LIST_DIR = 'traffic-signs-data'

combos = [(.001,.5,128)]

d = {0: 'Speed limit (20km/h)',
1: 'Speed limit (30km/h)',
2: 'Speed limit (50km/h)',
3: 'Speed limit (60km/h)',
4: 'Speed limit (70km/h)',
5: 'Speed limit (80km/h)',
6: 'End of speed limit (80km/h)',
7: 'Speed limit (100km/h)',
8: 'Speed limit (120km/h)',
9: 'No passing',
10: 'No passing for vehicles over 3.5 metric tons',
11: 'Right-of-way at the next intersection',
12: 'Priority road',
13: 'Yield',
14: 'Stop',
15: 'No vehicles',
16: 'Vehicles over 3.5 metric tons prohibited',
17: 'No entry',
18: 'General caution',
19: 'Dangerous curve to the left',
20: 'Dangerous curve to the right',
21: 'Double curve',
22: 'Bumpy road',
23: 'Slippery road',
24: 'Road narrows on the right',
25: 'Road work',
26: 'Traffic signals',
27: 'Pedestrians',
28: 'Children crossing',
29: 'Bicycles crossing',
30: 'Beware of ice/snow',
31: 'Wild animals crossing',
32: 'End of all speed and passing limits',
33: 'Turn right ahead',
34: 'Turn left ahead',
35: 'Ahead only',
36: 'Go straight or right',
37: 'Go straight or left',
38: 'Keep right',
39: 'Keep left',
40: 'Roundabout mandatory',
41: 'End of no passing',
42: 'End of no passing by vehicles over 3.5 metric tons'
}

if __name__ == '__main__':
	data = load_data()

	# eda
	print_dimensions(data)
	perform_eda(data) # -> check img folders for plots

	x_train, y_train, x_valid, y_valid, x_test, y_test, x_new, y_new = data

	tf.reset_default_graph()


	# Declare tensors
	x = tf.placeholder(tf.float32, [None, IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH])
	y = tf.placeholder(tf.int32, (None))
	one_hot_y = tf.one_hot(y,N_CLASSES)

	keep_prob = tf.placeholder(tf.float32)
	learning_rate = tf.placeholder(tf.float32)

	logits = covnet(x, keep_prob)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = one_hot_y))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
	accuracy_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Train model
	best_accuracy = 0
	best_combo = None

	saver = tf.train.Saver()

	for combo in combos:
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			num_examples = x_train.shape[0]
			print('COMBO: ', combo)
			learning_rate_val, keep_prob_val, batch_size_val = combo
			num_batches = num_examples // batch_size_val + 1
			for epoch in range(EPOCHS):
				x_train, y_train = shuffle(x_train, y_train)
				batches_qbar = tqdm(range(num_batches), desc='Epoch {}/{}'.format(epoch+1, EPOCHS), unit='batches')

				for batch in batches_qbar:
				# for batch in range(num_batches):
					start = batch * batch_size_val
					end = (batch + 1) * batch_size_val
					batch_x, batch_y = x_train[start:end], y_train[start:end]

					_, batch_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_val, learning_rate: learning_rate_val})

				print('  EPOCH: ', epoch+1)
				training_accuracy = evaluate(x_train, y_train, batch_size_val)
				print("  Training Accuracy   = {:.3f}".format(training_accuracy))
				validation_accuracy = evaluate(x_valid, y_valid, batch_size_val)
				print("  Validation Accuracy = {:.3f}".format(validation_accuracy))

			print('=====')
			if best_accuracy < validation_accuracy:
				best_accuracy = validation_accuracy
				best_combo = combo
				saver.save(sess, 'traffic')

	print('Best parameters are: ', best_combo, ' with accuracy of: ', best_accuracy)
	final_model_evaluate(best_combo[2])
