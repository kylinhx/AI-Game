from ultralytics import YOLO


MYMODEL_PATH = 'D:\\Work_space\\Projects\\AI-Game\\The_King_of_Fighters_XV\\model\\best.pt'

if __name__ == "__main__":

    model = YOLO(MYMODEL_PATH)

    results = model("./yolo/example/frame1870.jpg")
