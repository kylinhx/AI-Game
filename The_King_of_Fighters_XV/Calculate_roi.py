
# 定义窗口长宽
w = 1050
h = 720

# self_blood长宽
w_self_blood = 481-88
h_self_blood = 125-121
x_self_blood = 88
y_self_blood= 121

# enemy_blood长宽
w_enemy_blood = 961-568
h_enemy_blood = 125-121
x_enemy_blood = 568
y_enemy_blood = 121

# self_defind长宽
w_self_defend= 445-292
h_self_defend= 145-142
x_self_defend= 292
y_self_defend= 142

# enemy_defind长宽
w_enemy_defend= 759-606
h_enemy_defend= 145-142
x_enemy_defend= 606
y_enemy_defend= 142

# self_energy_number长宽
w_self_energy_number= 91 - 51
h_self_energy_number= 655 - 597
x_self_energy_number= 51
y_self_energy_number= 597

# enemy_energy_number长宽
w_enemy_energy_number= 1001 - 961
h_enemy_energy_number= 655 - 597
x_enemy_energy_number= 961
y_enemy_energy_number= 597

# self_energy长宽
w_self_energy= 263 - 99
h_self_energy= 631 - 626
x_self_energy= 99
y_self_energy= 626

# enemy_energy长宽
w_enemy_energy= 950 - 789
h_enemy_energy= 631 - 626
x_enemy_energy= 789
y_enemy_energy= 626

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
    