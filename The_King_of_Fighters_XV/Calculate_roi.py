
# 定义窗口长宽
w = 990
h = 560

# self_blood长宽
w_self_blood = 460-115
h_self_blood = 78-72
x_self_blood = 115
y_self_blood = 72

# enemy_blood长宽
w_enemy_blood = 879 - 534
h_enemy_blood = 78 - 72
x_enemy_blood = 534
y_enemy_blood = 72

# self_defind长宽
w_self_defend= 440-299
h_self_defend= 98-96
x_self_defend= 299
y_self_defend= 96

# enemy_defind长宽
w_enemy_defend= 724-583
h_enemy_defend= 98-96
x_enemy_defend= 583
y_enemy_defend= 96

# self_energy_number长宽
w_self_energy_number= 114 - 80
h_self_energy_number= 543 - 492
x_self_energy_number= 80
y_self_energy_number= 492

# enemy_energy_number长宽
w_enemy_energy_number= 912-879
h_enemy_energy_number= 542-493
x_enemy_energy_number= 879
y_enemy_energy_number= 493

# self_energy长宽
w_self_energy= 271 - 127
h_self_energy= 545 - 538
x_self_energy= 127
y_self_energy= 538

# enemy_energy长宽
w_enemy_energy= 897 - 751
h_enemy_energy= 545 - 538
x_enemy_energy= 751
y_enemy_energy= 538

def calculate_roi():
    return

if __name__ == "__main__":
    # caculate relative roi
    # [w, h, xmin, ymin]
    self_blood_roi = [w_self_blood / w, h_self_blood / h, x_self_blood / w, y_self_blood / h]
    
    enemy_blood_roi = [w_enemy_blood / w, h_enemy_blood / h, x_enemy_blood / w, y_enemy_blood / h]
    self_defend_roi = [w_self_defend / w, h_self_defend / h, x_self_defend / w, y_self_defend / h]
    enemy_defend_roi = [w_enemy_defend / w, h_enemy_defend / h, x_enemy_defend / w, y_enemy_defend / h]
    self_energy_roi = [w_self_energy / w, h_self_energy / h, x_self_energy / w, y_self_energy / h]
    self_energy_number_roi = [w_self_energy_number / w, h_self_energy_number / h, x_self_energy_number / w, y_enemy_energy_number / h]
    enemy_energy_roi = [w_enemy_energy / w, h_enemy_energy / h, x_enemy_energy / w, y_enemy_energy / h]
    enemy_energy_number_roi = [w_enemy_energy_number / w, h_enemy_energy_number / h, x_enemy_energy_number / w, y_enemy_energy_number / h]

    print("self_blood_roi: ", self_blood_roi)
    print("enemy_blood_roi: ", enemy_blood_roi)
    print("self_defend_roi: ", self_defend_roi)
    print("enemy_defend_roi: ", enemy_defend_roi)
    print("self_energy_roi: ", self_energy_roi)
    print("self_energy_number_roi: ", self_energy_number_roi)
    print("enemy_energy_roi: ", enemy_energy_roi)
    print("enemy_energy_number_roi: ", enemy_energy_number_roi)
    