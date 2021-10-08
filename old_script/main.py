import tensorflow as tf#version 1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filenames = tf.train.match_filenames_once('./code/pythonscript/neural_net/accelerometer/*.csv')
count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)
reader = tf.WholeFileReader()
filename, file_contents = reader.read(filename_queue)

def get_dataset(sess):
    sess.run(tf.local_variables_initializer())
    num_files = sess.run(count_num_files)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    accel_files = []
    xs = []
    for i in range(num_files):
        accel_file = sess.run(filename)
        accel_file_frame = pd.read_csv(accel_file, header=None, sep=',',
        names = ["Time", "XPos", "YPos", "ZPos"])
        accel_files.append(accel_file_frame)
        print(accel_file)
        x = [extract_feature_vector(sess, accel_file_frame.values)]
        x = np.matrix(x)
        if len(xs) == 0:
            xs = x
        else:
            xs = np.vstack((xs, x))
    return xs 


jerk = tf.placeholder(tf.float32)
max_jerk = tf.reduce_max(jerk, keepdims=True, axis=1)

def extract_feature_vector(sess, x):
 x_s = x[:, 1:]

 v_X = np.diff(x_s, axis=0)
 a_X = np.diff(v_X, axis=0)
 j_X = np.diff(a_X, axis=0)

 X = j_X
 mJerk = sess.run(max_jerk, feed_dict = {jerk: X})
 num_samples, num_features = np.shape(X)
 hist, bins = np.histogram(mJerk, bins=range(num_features + 1))
 return hist.astype(float) / num_samples

labels=['X', 'Y', 'Z']
fig, ax = plt.subplots()
ind = np.arange(len(labels))

width = 0.015
plots = []
colors = [np.random.rand(3,1).flatten() for i in range(num_samples)]
for i in range(num_samples):
    Xs = np.asarray(X[i]).reshape(-1)
    p = ax.bar(ind + i*width, Xs, width, color=colors[i])
    plots.append(p[0])
xticks = ind + width / (num_samples)
print(xticks)
ax.legend(tuple(plots), tuple(['P'+str(i+1) for i in range(num_samples)]),ncol=4)
ax.yaxis.set_units(inch)
ax.autoscale_view()
ax.set_xticks(xticks)
ax.set_xticklabels(labels)
ax.set_ylabel('Normalized jerk count')
ax.set_xlabel('Position (X, Y, Z)')
ax.set_title('Normalized jerk magnitude counts for Various Participants')
plt.show()

k = 2
max_iterations = 100
def initial_cluster_centroids(X, k):
    return X[0:k, :]

def assign_cluster(X, centroids):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    return mins

def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)
    return sums / counts

groups = None

with tf.Session() as sess:
    X = get_dataset(sess)
    centroids = initial_cluster_centroids(X, k)
    i, converged = 0, False
    while not converged and i < max_iterations:
        i += 1
        Y = assign_cluster(X, centroids)
        centroids = sess.run(recompute_centroids(X, Y))
    print(centroids)
    groups = Y.eval()
    print(groups) 

plt.scatter([X[:, 0]], [X[:, 1]], c=groups, s=50, alpha=0.5)
plt.plot([centroids[:, 0]], [centroids[:, 1]], 'kx', markersize=15)
plt.show()

segment_size = 50
def get_accel_data(accel_file):
    accel_file_frame = pd.read_csv(accel_file, header=None, sep=',',
    names = ["Time", "XPos", "YPos", "ZPos"])
    return accel_file_frame.values
def get_dataset_segmented(sess, accel_file):
    accel_data = get_accel_data(accel_file)
    print('accel_data', np.shape(accel_data))
    accel_length = np.shape(accel_data)[0]
    print('accel_length', accel_length)
    xs = []

    for i in range(accel_length / segment_size):
        accel_segment = accel_data[i*segment_size:(i+1)*segment_size, :]
        x = extract_feature_vector(sess, accel_segment)
        x = np.matrix(x)
        if len(xs) == 0:
            xs = x
        else:
            xs = np.vstack((xs, x))
    return accel_data, xs 


k = 5
with tf.Session() as sess:
    tf.global_variables_initializer()
    accel_data, X1 = get_dataset_segmented(sess, "./code/pythonscript/neural_net/Walking Activity/11.csv")
    centroids = initial_cluster_centroids(X1, k)
    i, converged = 0, False
    while not converged and i < max_iterations:
        i += 1
        Y1 = assign_cluster(X1, centroids)
        centroids = sess.run(recompute_centroids(X1, Y1))
        if i % 50 == 0:
            print('iteration', i)
    segments = sess.run(Y1)
    print('Num segments ', str(len(segments)))
    for i in range(len(segments)):
        seconds = (i * segment_size) / float(10)
        seconds = accel_data[(i * segment_size)][0]
        min, sec = divmod(seconds, 60)
        time_str = '{}m {}s'.format(min, sec)
        print(time_str, segments[i])

plt.scatter([X1[:, 0]], [X1[:, 1]], c=segments, s=50, alpha=0.5)
plt.plot([centroids[:, 0]], [centroids[:, 1]], 'kx', markersize=15)
plt.show()