def convert2_ue_path_flight_log(input_path=''):
    log_file = open(input_path, 'r')
    cc_file = open(input_path.split('.')[0]+'_cc.txt', 'w')
    lines = log_file.readlines()

    for line in lines:
        parts = line.split(',')  # UE
        image_name = parts[0].split('.')[0][-4:]
        cc_file.write(image_name+'.jpg,')  # cc
        cc_file.write(str(-float(parts[1])/100)+',')
        cc_file.write(str(float(parts[2])/100)+',')
        cc_file.write(str(float(parts[3])/100)+',')
        cc_file.write(str(-float(parts[4]))+',')
        cc_file.write(str(float(parts[5]))+',')
        cc_file.write(str(float(parts[6].strip('\n')))+'\n')
    cc_file.close()


def convert2_ue_path_flight_log_2(input_path=''):
    log_file = open(input_path, 'r')
    cc_file = open(input_path.split('.')[0]+'_cc5.txt', 'w')
    lines = log_file.readlines()

    for line in lines:
        if line != '\n':
            parts = line.split(',')  # UE
            image_name = parts[0][:-4]
            cc_file.write(image_name+'.jpg,')  # cc
            # cc_file.write(str(parts[0])+',')
            cc_file.write(str(-float(parts[1]))+',')
            cc_file.write(str(-float(parts[2]))+',')
            cc_file.write(str(float(parts[3]))+',')
            cc_file.write(str(float(parts[4]))+',')
            cc_file.write(str(float(parts[5]))+',')
            cc_file.write(str(float(parts[6].strip('\n')))+'\n')
    cc_file.close()


def convert2_ue_path_flight_log_final(input_path=''):
    log_file = open(input_path, 'r')
    cc_file = open(input_path[:-4]+'_cc.txt', 'w')
    lines = log_file.readlines()

    for line in lines:
        if line != '\n':
            parts = line.split(',')  # UE
            # image_name = parts[0][:-4]
            # cc_file.write(image_name+'.jpg,')  # cc
            cc_file.write(str(parts[0])+',')
            cc_file.write(str(float(parts[1]))+',')
            cc_file.write(str(-float(parts[2]))+',')                   # -y
            cc_file.write(str(float(parts[3]))+',')
            cc_file.write(str(float(parts[4]))+',')
            cc_file.write(str(float(parts[5]))+',')
            cc_file.write(str(float(parts[6].strip('\n'))+90)+'\n')    # yaw -90
    cc_file.close()


# convert2_ue_path_flight_log_final(r'E:\BaiduNetdiskDownload\UrbanScene3D-Dataset\Path\Bridge\Oblique.log')
convert2_ue_path_flight_log_final('cv-newyork-0.8-0.7-250/uav_position.txt')