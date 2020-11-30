import sqlite3
import statistics
import math
from scipy.stats import kurtosis
from scipy.stats import skew 
from scipy.fftpack import fft
from scipy.stats import entropy
import numpy as np
import sys

TIME_SLICE = 50

connection = sqlite3.connect('sensorsManager.db')
cursor = connection.cursor()

cursor.execute('DROP TABLE feature_vectors')
cursor.execute('''CREATE TABLE feature_vectors
                  (label text, 
                    max_ax float, min_ax float, mean_ax float, variance_ax float, kurtosis_ax float, skewness_ax float, pkpksignal_ax float, pkpktime_ax float, pkpkslope_ax float, alar_ax float, energy_ax float, entropy_ax float,
                    max_ay float, min_ay float, mean_ay float, variance_ay float, kurtosis_ay float, skewness_ay float, pkpksignal_ay float, pkpktime_ay float, pkpkslope_ay float, alar_ay float, energy_ay float, entropy_ay float,
                    max_az float, min_az float, mean_az float, variance_az float, kurtosis_az float, skewness_az float, pkpksignal_az float, pkpktime_az float, pkpkslope_az float, alar_az float, energy_az float, entropy_az float,
                    max_am float, min_am float, mean_am float, variance_am float, kurtosis_am float, skewness_am float, pkpksignal_am float, pkpktime_am float, pkpkslope_am float, alar_am float, energy_am float, entropy_am float,
                    max_gx float, min_gx float, mean_gx float, variance_gx float, kurtosis_gx float, skewness_gx float, pkpksignal_gx float, pkpktime_gx float, pkpkslope_gx float, alar_gx float, energy_gx float, entropy_gx float,
                    max_gy float, min_gy float, mean_gy float, variance_gy float, kurtosis_gy float, skewness_gy float, pkpksignal_gy float, pkpktime_gy float, pkpkslope_gy float, alar_gy float, energy_gy float, entropy_gy float,
                    max_gz float, min_gz float, mean_gz float, variance_gz float, kurtosis_gz float, skewness_gz float, pkpksignal_gz float, pkpktime_gz float, pkpkslope_gz float, alar_gz float, energy_gz float, entropy_gz float,
                    max_gm float, min_gm float, mean_gm float, variance_gm float, kurtosis_gm float, skewness_gm float, pkpksignal_gm float, pkpktime_gm float, pkpkslope_gm float, alar_gm float, energy_gm float, entropy_gm float,
                    max_mx float, min_mx float, mean_mx float, variance_mx float, kurtosis_mx float, skewness_mx float, pkpksignal_mx float, pkpktime_mx float, pkpkslope_mx float, alar_mx float, energy_mx float, entropy_mx float,
                    max_my float, min_my float, mean_my float, variance_my float, kurtosis_my float, skewness_my float, pkpksignal_my float, pkpktime_my float, pkpkslope_my float, alar_my float, energy_my float, entropy_my float,
                    max_mz float, min_mz float, mean_mz float, variance_mz float, kurtosis_mz float, skewness_mz float, pkpksignal_mz float, pkpktime_mz float, pkpkslope_mz float, alar_mz float, energy_mz float, entropy_mz float,
                    max_mm float, min_mm float, mean_mm float, variance_mm float, kurtosis_mm float, skewness_mm float, pkpksignal_mm float, pkpktime_mm float, pkpkslope_mm float, alar_mm float, energy_mm float, entropy_mm float
                    )''')

# get the number of actions
cursor.execute("SELECT * FROM data WHERE item = 'num_recorded'")
num_actions = cursor.fetchone()[1]

for action_id in range(num_actions):
    t = (str(action_id),)
    cursor.execute("SELECT * FROM actions WHERE id=?", t)
    action = cursor.fetchone()
    print(action)
    num_readings = action[4]

    if action[3] == 'holding' or action[0] == 2:
        label = 'holding'
    else:
        label = 'not holding'
        # num_readings = int(num_readings/6.28296547821)

    acc_readings = []
    for row in cursor.execute("SELECT * FROM acc WHERE id=?", t):
        acc_readings.append(row)

    mag_readings = []
    for row in cursor.execute("SELECT * FROM mag WHERE id=?", t):
        mag_readings.append(row)

    gyro_readings = []
    for row in cursor.execute("SELECT * FROM gyro WHERE id=?", t):
        gyro_readings.append(row)

    # skip the first 5 seconds of data
    start_value = 500

    # skip the last 5 seconds of data
    end_value = num_readings - 500

    # slice the data and make stats of each slice
    saved_readings = []
    saved_stats = []
    print((end_value - start_value)/TIME_SLICE)
    for reading in range(start_value, end_value):
        if reading % TIME_SLICE == 0 and len(saved_readings) == TIME_SLICE:
            acc_x_v = []
            acc_y_v = []
            acc_z_v = []
            acc_m_v = []

            gyro_x_v = []
            gyro_y_v = []
            gyro_z_v = []
            gyro_m_v = []

            mag_x_v = []
            mag_y_v = []
            mag_z_v = []
            mag_m_v = []

            for vals in saved_readings:
                acc_x_v.append(vals[0][2])
                acc_y_v.append(vals[0][3])
                acc_z_v.append(vals[0][4])
                acc_m_v.append(math.sqrt(pow(vals[0][2],2) + pow(vals[0][3],2) + pow(vals[0][4],2)))

                gyro_x_v.append(vals[1][2])
                gyro_y_v.append(vals[1][3])
                gyro_z_v.append(vals[1][4])
                gyro_m_v.append(math.sqrt(pow(vals[1][2],2) + pow(vals[1][3],2) + pow(vals[1][4],2)))

                mag_x_v.append(vals[2][2])
                mag_y_v.append(vals[2][3])
                mag_z_v.append(vals[2][4])
                mag_m_v.append(math.sqrt(pow(vals[2][2],2) + pow(vals[2][3],2) + pow(vals[2][4],2)))

            def get_stats(array_vals):
                mi = array_vals[0]
                ma = array_vals[0]
                mi_i = 0
                ma_i = 0
                for i in range(1, len(array_vals)): 
                    if array_vals[i] < mi: 
                        mi = array_vals[i] 
                        mi_i = i
                    if array_vals[i] > ma: 
                        ma = array_vals[i]
                        ma_i = i

                mean = statistics.fmean(array_vals)
                var = statistics.pvariance(array_vals)
                k = kurtosis(array_vals)
                skewness = skew(array_vals)
                ppsv = ma - mi
                ppti = ma_i + mi_i
                if ppti != 0:
                    ppsl = ppsv / ppti
                else:
                    ppsl = sys.maxsize
                
                if ma != 0:
                    alar = abs(ma_i / ma)
                else:
                    alar = sys.maxsize

                fourier = fft(array_vals)
                energy = 0
                for val in fourier:
                    energy = abs(val)
                energy = pow(energy, 2)

                value,counts = np.unique(fourier, return_counts=True)
                entrop = entropy(counts, base=2)

                return (ma, mi, mean, var, k, skewness, ppsv, ppti, ppsl, alar, energy, entrop)
            
            # get stats
            acc_x_s = get_stats(acc_x_v)
            acc_y_s = get_stats(acc_y_v)
            acc_z_s = get_stats(acc_z_v)
            acc_m_s = get_stats(acc_m_v)
            gyro_x_s = get_stats(gyro_x_v)
            gyro_y_s = get_stats(gyro_y_v)
            gyro_z_s = get_stats(gyro_z_v)
            gyro_m_s = get_stats(gyro_m_v)
            mag_x_s = get_stats(mag_x_v)
            mag_y_s = get_stats(mag_y_v)
            mag_z_s = get_stats(mag_z_v)
            mag_m_s = get_stats(mag_m_v)

            # save tupl
            saved_stats.append((label, acc_x_s[0], acc_x_s[1], acc_x_s[2], acc_x_s[3], acc_x_s[4], acc_x_s[5], acc_x_s[6], acc_x_s[7], acc_x_s[8], acc_x_s[9], acc_x_s[10], acc_x_s[11], \
                                       acc_y_s[0], acc_y_s[1], acc_y_s[2], acc_y_s[3], acc_y_s[4], acc_y_s[5], acc_y_s[6], acc_y_s[7], acc_y_s[8], acc_y_s[9], acc_y_s[10], acc_y_s[11], \
                                       acc_z_s[0], acc_z_s[1], acc_z_s[2], acc_z_s[3], acc_z_s[4], acc_z_s[5], acc_z_s[6], acc_z_s[7], acc_z_s[8], acc_z_s[9], acc_z_s[10], acc_z_s[11], \
                                       acc_m_s[0], acc_m_s[1], acc_m_s[2], acc_m_s[3], acc_m_s[4], acc_m_s[5], acc_m_s[6], acc_m_s[7], acc_m_s[8], acc_m_s[9], acc_m_s[10], acc_m_s[11], \

                                       gyro_x_s[0], gyro_x_s[1], gyro_x_s[2], gyro_x_s[3], gyro_x_s[4], gyro_x_s[5], gyro_x_s[6], gyro_x_s[7], gyro_x_s[8], gyro_x_s[9], gyro_x_s[10], gyro_x_s[11], \
                                       gyro_y_s[0], gyro_y_s[1], gyro_y_s[2], gyro_y_s[3], gyro_y_s[4], gyro_y_s[5], gyro_y_s[6], gyro_y_s[7], gyro_y_s[8], gyro_y_s[9], gyro_y_s[10], gyro_y_s[11], \
                                       gyro_z_s[0], gyro_z_s[1], gyro_z_s[2], gyro_z_s[3], gyro_z_s[4], gyro_z_s[5], gyro_z_s[6], gyro_z_s[7], gyro_z_s[8], gyro_z_s[9], gyro_z_s[10], gyro_z_s[11], \
                                       gyro_m_s[0], gyro_m_s[1], gyro_m_s[2], gyro_m_s[3], gyro_m_s[4], gyro_m_s[5], gyro_m_s[6], gyro_m_s[7], gyro_m_s[8], gyro_m_s[9], gyro_m_s[10], gyro_m_s[11], \

                                       mag_x_s[0], mag_x_s[1], mag_x_s[2], mag_x_s[3], mag_x_s[4], mag_x_s[5], mag_x_s[6], mag_x_s[7], mag_x_s[8], mag_x_s[9], mag_x_s[10], mag_x_s[11], \
                                       mag_y_s[0], mag_y_s[1], mag_y_s[2], mag_y_s[3], mag_y_s[4], mag_y_s[5], mag_y_s[6], mag_y_s[7], mag_y_s[8], mag_y_s[9], mag_y_s[10], mag_y_s[11], \
                                       mag_z_s[0], mag_z_s[1], mag_z_s[2], mag_z_s[3], mag_z_s[4], mag_z_s[5], mag_z_s[6], mag_z_s[7], mag_z_s[8], mag_z_s[9], mag_z_s[10], mag_z_s[11], \
                                       mag_m_s[0], mag_m_s[1], mag_m_s[2], mag_m_s[3], mag_m_s[4], mag_m_s[5], mag_m_s[6], mag_m_s[7], mag_m_s[8], mag_m_s[9], mag_m_s[10], mag_m_s[11], \
            ))
            saved_readings.clear()

        saved_readings.append((acc_readings[reading], mag_readings[reading], gyro_readings[reading]))
    
    cursor.executemany('''INSERT INTO feature_vectors VALUES (?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?,
    ?,?,?,?,?,?,?,?,?,?,?,?)
    ''', saved_stats)

connection.commit()
connection.close()