from matplotlib import pyplot as plt
import h5py


all_labels = []
all_x = []
all_y = []
all_z = []
with h5py.File('E:/Datasets/CS 4440 Final Project/mat_files_full/26_M4_F5_cropped_data_v2.mat', 'r') as f:
    cropped_mocap = f['cropped_data']['cropped_mocap'][()]
    trial_data = f[cropped_mocap[0][0]]
    raw_labels = trial_data['Labels']
    raw_data = trial_data['Loc']
    for label_index in range(len(raw_labels)):
        label_arr = f[raw_labels[label_index][0]][()]
        label = ''.join([chr(x[0]) for x in label_arr])
        label_data = f[raw_data[label_index][0]][()]
        label_x = label_data[0][100]
        label_y = label_data[1][100]
        label_z = label_data[2][100]
        all_labels.append(label)
        all_x.append(label_x)
        all_y.append(label_y)
        all_z.append(label_z)

sub1_labels = []
sub1_x = []
sub1_y = []
sub1_z = []
sub2_labels = []
sub2_x = []
sub2_y = []
sub2_z = []
ob_labels = []
ob_x = []
ob_y = []
ob_z = []
for i in range(len(all_labels)):
    if 'Sub1' in all_labels[i]:
        sub1_labels.append(all_labels[i])
        sub1_x.append(all_x[i])
        sub1_y.append(all_y[i])
        sub1_z.append(all_z[i])
    if 'Sub2' in all_labels[i]:
        sub2_labels.append(all_labels[i])
        sub2_x.append(all_x[i])
        sub2_y.append(all_y[i])
        sub2_z.append(all_z[i])
    if 'OB' in all_labels[i]:
        ob_labels.append(all_labels[i])
        ob_x.append(all_x[i])
        ob_y.append(all_y[i])
        ob_z.append(all_z[i])


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(sub1_x, sub1_y, sub1_z)
for i, label in enumerate(sub1_labels):
    ax1.text(sub1_x[i], sub1_y[i], sub1_z[i], label)
ax1.set_title('Sub 1 wide')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(sub1_x, sub1_y, sub1_z)
ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 2])
ax2.set_zlim([0, 2])
ax2.set_title('Sub 1 proportional')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(sub2_x, sub2_y, sub2_z)
for i, label in enumerate(sub2_labels):
    ax3.text(sub2_x[i], sub2_y[i], sub2_z[i], label)
ax3.set_title('Sub 2 wide')

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter(sub2_x, sub2_y, sub2_z)
ax4.set_xlim([-1, 1])
ax4.set_ylim([0, 2])
ax4.set_zlim([0, 2])
ax4.set_title('Sub 2 proportional')

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(ob_x, ob_y, ob_z)
for i, label in enumerate(ob_labels):
    ax5.text(ob_x[i], ob_y[i], ob_z[i], label)
ax5.set_title('Object wide')

plt.show()
