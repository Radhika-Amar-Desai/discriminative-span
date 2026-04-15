import os

dir = r"C:\Users\97433\Knowing_the_difference\data\datasets\skin_lesion\final_real_data\healthy"
length = len(os.listdir(dir))

print(f"The length is: {length}")

dir = r"C:\Users\97433\Knowing_the_difference\data\datasets\skin_lesion\final_real_data\lesion"
length = len(os.listdir(dir))

print(f"The length is: {length}")