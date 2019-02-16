#!/usr/bin/env python

#Name: Harshal Ganesh Jagtap
#UB Person No: 50290606
# Email ID: harshalg@buffalo.edu


import rosbag
import math
import tf
import time
import numpy as np
import rospy
import os
def discretize(x, y, heading):
    row = (x / 20) + 1
    col = (y / 20) + 1
    theta = (heading / discreteSize) + 1
    return int(row), int(col), int(theta)


def inverseDiscretize(X, Y, theta):
    xc = X * 20
    yc = Y * 20
    tc = theta * discreteSize
    return xc, yc, tc


def incorporateNoise(dPoint, mean, variance):
    gauss = (1 / math.sqrt(2 * math.pi * (variance ** 2))) * math.exp(-((dPoint - mean) ** 2) / (2 * variance ** 2))
    return gauss


def calculate_trajectory(x1, y1, theta1, x2, y2, theta2):
    deltaTrans = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 20
    deltaTheta1 = math.degrees(math.atan2((y2 - y1), (x2 - x1))) - theta1
    deltaTheta2 = theta2 - math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    return deltaTheta1, deltaTrans, deltaTheta2

print("Bayes filter started.")
startTime = time.time()
rospy.init_node("bayes_filter",anonymous=True)
grid_path = str(os.path.dirname(os.path.realpath(__file__))) + "/grid.bag"
grid = rosbag.Bag(grid_path)
grid_data = grid.read_messages(topics=['Movements', 'Observations'])
model_data = []
notebook = open("trajectory.txt","a")
discreteSize = 90  # Discretization sizeup
tags = [[125, 525], [125, 325], [125, 125], [425, 125], [425, 325], [425, 525]]
startPos = [12 * 20, 28 * 20, 200.52]
current_pose = [12, 28, int((200.52 / discreteSize) + 1)]

samples = []
for data in grid_data:
    samples.append(data)

belief = np.zeros((35, 35, 360 / discreteSize), dtype=np.float64)
belief[current_pose[0]-1, current_pose[1]-1, current_pose[2]-1] = 1.0
final_points = []

for sample in samples:

    next_belief = np.zeros((35, 35, 360 / discreteSize), dtype=np.float64)
    if (sample[0] == "Movements"):
        rot1 = math.degrees(tf.transformations.euler_from_quaternion([sample[1].rotation1.x,
                                                         sample[1].rotation1.y,
                                                         sample[1].rotation1.z,
                                                         sample[1].rotation1.w])[2])
        rot2 = math.degrees(tf.transformations.euler_from_quaternion([sample[1].rotation2.x,
                                                         sample[1].rotation2.y,
                                                         sample[1].rotation2.z,
                                                         sample[1].rotation2.w])[2])
        translation = (sample[1].translation)*100
        for i in range(1, 36):
            for j in range(1, 36):
                for k in range(1, (360 / discreteSize)+1):
                    prob = np.zeros((35, 35, 360 / discreteSize), dtype=np.float64)
                    for l in range(1, 36):
                        for m in range(1, 36):
                            for n in range(1, (360 / discreteSize)+1):
                                theta1, trans, theta2 = calculate_trajectory(i, j, k*discreteSize, l, m, n*discreteSize)
                                p1 = incorporateNoise(theta1, rot1, discreteSize / 2)
                                p2 = incorporateNoise(trans, translation, 10)
                                p3 = incorporateNoise(theta2, rot2, discreteSize / 2)
                                prob[l - 1][m - 1][n - 1] = p1 * p2 * p3
                    next_belief[i - 1][j - 1][k - 1] = np.sum(np.multiply(prob, belief))
        belief = next_belief.copy()
    else:
        range_var = (sample[1].range)*100
        bearing = tf.transformations.euler_from_quaternion([sample[1].bearing.x,
                                                            sample[1].bearing.y,
                                                            sample[1].bearing.z,
                                                            sample[1].bearing.w])
        bearing = math.degrees(bearing[2])
        landmark = tags[sample[1].tagNum]
        landmark = [int(landmark[0] /20) + 1, int(landmark[1] /20) + 1]
        for u in range(1, 36):
            for v in range(1, 36):
                for w in range(1, (360 / discreteSize)+1):
                    delta_range = math.sqrt((landmark[0] - u) ** 2 + (landmark[1] - v) ** 2) * 20
                    delta_bearing = math.degrees(math.atan2(landmark[1] - v , landmark[0] - u)) - (w * discreteSize)
                    p11 = incorporateNoise(delta_range, range_var, 10)
                    p22 = incorporateNoise(delta_bearing, bearing, discreteSize / 2)
                    p12 = p11 * p22
                    next_belief[u - 1][v - 1][w - 1] = p12 * belief[u - 1][v - 1][w - 1]
        belief = next_belief.copy()
        belief = np.divide(belief, np.sum(belief))
        trajectory = np.unravel_index(belief.argmax(),np.shape(belief))
        notebook.write("Probability "+str(np.max(belief))+" Calculated pose " + str([trajectory[0]+1,trajectory[1]+1,trajectory[2]+1])+"\n")
        final_points.append([trajectory[0], trajectory[1], trajectory[2]])


totalTime = time.time() - startTime
print("Total time: " + str(totalTime / 60) + " minutes")
