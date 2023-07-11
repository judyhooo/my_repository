## Work Notes 2023.5.12

* 新建模型导入UE4时找不到红色材质

解决方法：在Maya中导入原模型，并在Maya中改变材质

![image-20230512150737925](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230512150737925.png)

在工具栏选中渲染选项，选择使用Lambert材质，同时在颜色区进行更改。完成后，导出为.fbx模型，注意要导出所有包含的东西。

在UE4里导入时，要注意在导入下拉选项中选择新建材质，这样就能在UE4里创建新的材质。

![image-20230512151034860](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230512151034860.png)

最后导入成功，在根据所需大小进行调整即可：

![image-20230512151214236](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230512151214236.png)

* 无人机无法飞过门框，是碰撞盒的问题，要修改碰撞选项。

解决方案：在物体details下拉框下，找到碰撞（collision）选项，在碰撞预设选项（Collision preset）下修改，默认为Default，修改成OverlapAll，这样无人机就能飞过门框了。

![image-20230512151927917](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230512151927917.png)

* 采用原论文使用的airsimdroneracingvae的相关接口，始终出现接口期望（Expected）参数和实际（got）返回参数不匹配的问题

![image-20230512152517909](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230512152517909.png)

核心问题：根本问题在于rpc协议通信的时候client端和server端传递的参数对不上，实际上该函数需要的参数是4个，但是server端返回了6个，导致这个问题的原因应该是编译对应的AirSim的版本不匹配，但是在降低了AirSim到airsimdroneracingvae官方文档匹配的版本后仍然出现问题。

#### 修改方案：直接基于airsim去进行数据采集。

1. 仅使用x,y,z的随机采样。

```python
import airsim
import os
import numpy as np

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取图像路径
folder_path = "D:/airsim_images/"  # 保存图像的文件夹路径
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 设置随机采样的范围和数量
num_samples = 1000  # 需要采样的数量
x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -20, -10  # 采样范围

# 随机采样并保存图像和位姿信息
poses = []
for i in range(num_samples):
    # 随机生成目标位置，并设置姿态朝向正向
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    poses.append(pose)

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0].image_data_uint8

    # 保存图像和位姿信息
    img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.png".format(i, x, y, z)
    img_filepath = os.path.join(folder_path, img_filename)
    airsim.write_file(os.path.normpath(img_filepath), img_raw)

    pose_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.txt".format(i, x, y, z)
    pose_filepath = os.path.join(folder_path, pose_filename)
    with open(pose_filepath, 'w') as f:
        f.write("{0}\n".format(pose))

print("全部图像和位姿信息均已保存到文件夹：", folder_path)
```

直接运行改代码会出现两个问题：

>* 存储到文件夹中时会出现一张图片一个位姿关系的txt文件，不方便进行数据处理
>
>* 图片不能正常打开和显示，该问题出在使用client.simGetImages时获取到的图片格式是bytes，需要进行转换。

**转换方式**：

```python
    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # 将字节流转换为PIL的Image对象
    img1d = np.fromstring(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)
```

但是使用该方法会出现错误

`DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead   img1d = np.fromstring(img_raw.image_data_uint8, dtype=np.uint8)`

**问题：**

这个警告是由于PIL库中的`fromstring()`方法在处理二进制数据时有缺陷，Pillow库中已将其弃用。问题的解决方法是将`fromstring()`替换为`frombuffer()`

修改如下：

```python
import airsim
import os
import numpy as np

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取图像路径
folder_path = "E:/FunctionMethod/airsim_images"  # 保存图像的文件夹路径
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 设置随机采样的范围和数量
num_samples = 1000  # 需要采样的数量
x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -20, -10  # 采样范围

# 随机采样并保存图像和位姿信息
poses = []
for i in range(num_samples):
    # 随机生成目标位置，并设置姿态朝向正向
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    poses.append(pose)

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # get numpy array
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)

    # 保存图像和位姿信息
    img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.png".format(i, x, y, z)
    img_filepath = os.path.join(folder_path, img_filename)
    airsim.write_png(os.path.normpath(img_filepath), img_rgb1)

    pose_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.txt".format(i, x, y, z)
    pose_filepath = os.path.join(folder_path, pose_filename)
    with open(pose_filepath, 'w') as f:
        f.write("{0}\n".format(pose))

print("全部图像和位姿信息均已保存到文件夹：", folder_path)

```

**为了解决一张图片一个txt位姿文件这个不方便数据处理的问题：**使用`pandas`将位姿文件存储到一个csv文件中。

```python
import airsim
import os
import numpy as np
import pandas as pd

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取图像路径
folder_path = "E:/FunctionMethod/airsim_images"

# 保存位姿信息的空DataFrame
poses_df = pd.DataFrame(columns=['index', 'x', 'y', 'z'])

# 设置随机采样的范围和数量
num_samples = 10  # 需要采样的数量
x_min, x_max, y_min, y_max, z_min, z_max = -50, 50, -50, 50, -20, -10  # 采样范围

# 随机采样并保存图像和位姿信息
poses_list = []
for i in range(num_samples):
    # 随机生成目标位置，并设置姿态朝向正向

    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    poses_list.append({'index': i, 'x': x, 'y': y, 'z': z})

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # 将字节流转换为PIL的Image对象
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)

    # 保存PNG格式的图像
    img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}.png".format(i, x, y, z)
    img_filepath = os.path.join(folder_path, img_filename)
    airsim.write_png(os.path.normpath(img_filepath), img_rgb1)

print("全部图像和位姿信息均已保存到文件夹：", folder_path)

# 将位姿信息保存到csv文件中
poses_df = pd.DataFrame(poses_list)
poses_df.to_csv(os.path.join(folder_path, 'poses.csv'), index=False)

```

2. 完善的地方，如果采用上述代码，x,y,z的坐标是在世界范围内随机采取，有可能与gate相差太远，采集到的图片效果不好。
   * 解决方法1，较为麻烦。因为该场景下没有设置PlayerStart，因此AirSim模式下首次放置的无人机是根据手动拖动的场景位置而生成的。因此可以获取首次无人机的坐标位置，再来进行随机。

```python
import airsim
import numpy as np
import math
import csv
import os

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取对API的控制权
client.enableApiControl(True)

# 设置保存目录，确保目录存在
save_dir = "E:/FunctionMethod/airsim_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义生成随机点的最大和最小范围
x_min, x_max = -100, 100
y_min, y_max = -100, 100
z_min, z_max = -50, -30

num_samples = 1000  # 要生成的采样数量
rows = []  # 要保存到CSV中的数据

for i in range(num_samples):
    # 获取无人机当前状态
    state = client.getMultirotorState()
    x0 = state.kinematics_estimated.position.x_val
    y0 = state.kinematics_estimated.position.y_val
    z0 = state.kinematics_estimated.position.z_val
    yaw = math.degrees(airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2])

    # 生成随机偏移量
    dx = np.random.uniform(x_min, x_max)
    dy = np.random.uniform(y_min, y_max)
    dz = np.random.uniform(z_min, z_max)

    # 将偏移量转换为相对于无人机当前位置的坐标
    d_x, d_y = dx * math.cos(math.radians(yaw)) - dy * math.sin(math.radians(yaw)), dx * math.sin(math.radians(yaw)) + dy * math.cos(math.radians(yaw))
    x = x0 + d_x
    y = y0 + d_y
    z = z0 + dz

    print("Target position: (x: ", x, " y: ", y, " z: ", z, ")")

    # 将位姿保存到CSV列表
    row = [x, y, z, 0, 0, yaw]
    rows.append(row)

    # 构造位姿
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, yaw))

    # 获取相机图像
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    img_raw = responses[0]

    # get numpy array
    img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
    img_rgb1 = np.flipud(img_rgb)

    # 将图像保存到文件
    filename = f"sample_{i:04d}.png"
    filepath = os.path.join(save_dir, filename)
    airsim.write_png(os.path.normpath(filepath), img_rgb1)

    # 飞向目标位置
    client.moveToPositionAsync(pose.position.x_val, pose.position.y_val, pose.position.z_val, 5).join()

# release control
client.enableApiControl(False)
# 将位姿列表保存到CSV文件
with open(os.path.join(save_dir, "poses.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
```

​	该方法存在的问题是，每一次无人机的坐标获取都是在上一次的基础上，所以采集图像时有可能会朝着某个方向一直运动。

​	**解决方法2：**直接将gate的位置挪到世界地图的起始位置。

3. 改代码还存在一些量纲方面的问题，Python脚本代码中规定的x,y,z的范围默认情况下是“米”为单位。

![image-20230515003946930](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230515003946930.png)

​	而UE4中模型的details中的x,y,z坐标的单位是“厘米”，所以为了更好地进行图像采集，需要同时调整python脚本和UE4中的数值，使之数据采集图像更加合理。

![image-20230515004223805](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230515004223805.png)

![image-20230515004306306](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230515004306306.png)

4. 数据采集过程中得到的图像不包含gate，原因是`ImageRequest` 对象的第一个参数（相机名称）有偏差，因为运行AirSim后无人机实际上是侧对gate，可以采取两种方式去调整：

   * 调整gate位置，将其选装放置到无人机的正前方，再去调用无人机相机的前视相机来进行数据采集。

   * 修改`ImageRequest` 对象的第一个参数，采用左视相机。常用的相机及其对应的序号如下：

     >- 相机名称 "0": 前视相机
     >- 相机名称 "1": 右视相机
     >- 相机名称 "2": 后视相机
     >- 相机名称 "3": 下视相机
     >- 相机名称 "4": 左视相机

5. 仅使用x,y,z的话不够完整和全面，因此需要加入yaw,pitch,roll

```python
import airsim
import os
import numpy as np
import pandas as pd

# 连接到AirSim模拟器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取图像路径
folder_path = "E:/FunctionMethod/airsim_images"

# 保存位姿信息的空DataFrame
poses_df = pd.DataFrame(columns=['index', 'x', 'y', 'z', 'yaw', 'pitch', 'roll'])

# 设置随机采样的范围和数量
num_samples = 100  # 需要采样的数量
x_min, x_max, y_min, y_max, z_min, z_max = -4, 4, -4, 4, -5, -2  # 位置范围
yaw_min, yaw_max, pitch_min, pitch_max, roll_min, roll_max = -90, 90, -45, 45, -45, 45  # 姿态范围

# 相机列表
camera_list = ["0", "1", "2", "3", "4"]

# 随机采样并保存图像和位姿信息
poses_list = []
for i in range(num_samples):
    # 随机生成目标位置，并设置姿态朝向
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    yaw = np.random.uniform(yaw_min, yaw_max)
    pitch = np.random.uniform(pitch_min, pitch_max)
    roll = np.random.uniform(roll_min, roll_max)
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
    poses_list.append({'index': i, 'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': pitch, 'roll': roll})

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # # 获取相机图像
    # responses = client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
    # img_raw = responses[0]

    # 遍历相机列表，获取每个相机的图像
    for j, camera_name in enumerate(camera_list):
        # 获取相机图像
        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
        img_raw = responses[0]

        # 将字节流转换为PIL的Image对象
        img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
        img_rgb1 = np.flipud(img_rgb)

        # 保存PNG格式的图像
        img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}_yaw_{4:.2f}_pitch_{5:.2f}_roll_{6:.2f}_camera_{4}.png".format(i, x, y, z, yaw, pitch, roll, j)
        img_filepath = os.path.join(folder_path, img_filename)
        airsim.write_png(os.path.normpath(img_filepath), img_rgb1)

print("全部图像和位姿信息均已保存到文件夹：", folder_path)

# 将位姿信息保存到csv文件中
poses_df = pd.DataFrame(poses_list)
poses_df.to_csv(os.path.join(folder_path, 'poses.csv'), index=False)

```

## Work Notes 2023.5.16

1. 截出来的图像默认大小是256*144，需要对图片进行裁剪

* 先将图片裁剪成144*144（为了保留更多的gate信息，从中间裁剪）

```python
from PIL import Image
import os

# 设置裁剪后的图像大小
target_size = (120, 120)

# 设置文件夹路径
folder_path = "E:/FunctionMethod/rotate pictures/desert"

# 遍历文件夹中所有的png文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 打开源图像文件
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # 裁剪图像
        cropped_img = img.crop((56, 0, 200, 144))

        # 打开原始图片并裁剪
        save_path = os.path.join(folder_path, "cropped_" + filename)
        cropped_img.save(save_path, "PNG")

```

* 再将图片等比例resize到120*120

```PYTHON
from PIL import Image
import os

# 设置文件夹路径
folder_path = "E:/FunctionMethod/resized pictures/desert 144"

# 遍历文件夹中所有的png文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # 打开源图像文件
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # 裁剪图像
        resized_img = img.resize((120, 120), resample=Image.BICUBIC)

        # 打开原始图片并裁剪
        save_path = os.path.join(folder_path, "resized_" + filename)
        resized_img.save(save_path, "PNG")

```

![image-20230516155349447](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230516155349447.png)

2. 获取到含有x,y,z,yaw,pitch,roll的位姿信息csv文件后，直接根据该文件去进行截图。

```python
import airsim
import os
import csv
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()

# 设置相机和文件路径
camera_list = ["0", "1", "2", "3", "4"]
folder_path = "E:/FunctionMethod/airsim_images"

# 读取位姿信息文件（csv格式）
poses_csv_file = open("E:/FunctionMethod/airsim_images/poses test.csv", "r")
pos_reader = csv.DictReader(poses_csv_file)

# 循环采样并保存图像和位姿信息
for i, row in enumerate(pos_reader):
    # 获取姿态信息
    x, y, z = float(row['x']), float(row['y']), float(row['z'])
    yaw, pitch, roll = float(row['yaw']), float(row['pitch']), float(row['roll'])
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))

    # 移动到目标位置
    client.simSetVehiclePose(pose, True)
    airsim.time.sleep(1.0)

    # 遍历相机列表，获取每个相机的图像
    for j, camera_name in enumerate(camera_list):
        responses = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])
        img_raw = responses[0]

        # 将字节流转换为PIL的Image对象
        img1d = np.frombuffer(img_raw.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_raw.height, img_raw.width, 3)
        img_rgb1 = np.flipud(img_rgb)

        # 保存PNG格式的图像
        img_filename = "pose_{0}_x_{1:.2f}_y_{2:.2f}_z_{3:.2f}_yaw_{4:.2f}_pitch_{5:.2f}_roll_{6:.2f}_camera_{7}.png".format(i, x, y, z, yaw, pitch, roll, j)
        img_filepath = os.path.join(folder_path, img_filename)
        airsim.write_png(os.path.normpath(img_filepath), img_rgb1)

print("图像和位姿信息均已保存到文件夹：", folder_path)

```

3. 利用该位姿信息时需要更换场景，但是在更换到雪地场景时出现了一些问题，运行AirSim时无人机的初始位置在世界的最边缘，无法正常放置。

![image-20230516173745148](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230516173745148.png)
**解决方案：**在用户文档下找到AirSim，并将settings.json文件进行修改，设置无人机的初始位置。

![image-20230516173616150](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230516173616150.png)

4. 虽然更换场景且修改初始位置能够在该snow场景下成功地启动无人机，但始终会与另外一个场景下有差异，在之前的desert场景中，启动无人机会根据实际拖拽的位置来进行放置，在settings.json文件中去修改效果不大。但是换到另外一个snow场景中settings.json文件的设置又会生效。为了保持一致性，最好的方法是更换desert的表面材质为snow场景。

![image-20230517110141553](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230517110141553.png)

![image-20230517110750086](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230517110750086.png)

5. 更换场景后直接运行上述代码来进行相同位姿条件下的截图，但是这里仍然会出现一些问题，就是可能在飞行过程中的一些随机抖动，会导致截出来的图像有失偏颇。

![image-20230518102058586](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230518102058586.png)

## Work Notes 2023.5.18

1. 计算相同序号下的余弦相似度：

```python
import cv2
import matplotlib.pyplot as plt
import torchvision as torchvision
from PIL import Image
import numpy as np

print('----------------------相同序号余弦相似度--------------------')
f = open('analysis results/desert_snow_cosin(-1,1)_120.txt', 'w')
for i in range(0, 1000):
    if i < 10:
        img1 = cv2.imread('E:/FunctionMethod/resized pictures/desert 120/000' + str(i) + '.png')
        img2 = cv2.imread('E:/FunctionMethod/resized pictures/snow 120/000' + str(i) + '.png')
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
        img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write('desert与snow场景第' + str(i) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
    elif 10 <= i < 100:
        img1 = cv2.imread('E:/FunctionMethod/resized pictures/desert 120/00' + str(i) + '.png')
        img2 = cv2.imread('E:/FunctionMethod/resized pictures/snow 120/00' + str(i) + '.png')
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
        img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write('desert与snow场景第' + str(i) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
    else:
        img1 = cv2.imread('E:/FunctionMethod/resized pictures/desert 120/0' + str(i) + '.png')
        img2 = cv2.imread('E:/FunctionMethod/resized pictures/snow 120/0' + str(i) + '.png')
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1_rgb_flat = np.array(img1_rgb.ravel()).astype(int)
        img2_rgb_flat = np.array(img2_rgb.ravel()).astype(int)
        img1_rgb_flat = img1_rgb_flat / 255.0*2.0-1.0
        img2_rgb_flat = img2_rgb_flat / 255.0*2.0-1.0
        cos_simi = np.sum(img1_rgb_flat * img2_rgb_flat) / (np.linalg.norm(img1_rgb_flat) * np.linalg.norm(img2_rgb_flat))
        f.write('desert与snow场景第' + str(i) + '张图之间的余弦相似度,' + str(cos_simi) + '\n')
```

2. 绘制频率分布直方图并计算KL散度。

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.losses import kl_divergence
import numpy as np
from scipy.stats import entropy

reviews1 = pd.read_csv('analysis results/desert_snow_cosin(-1,1)_120.csv')
reviews3 = pd.read_csv('analysis results/z_desert_snow_cosin(-1,1).csv')

fig = plt.figure(figsize=(15, 12), dpi=100)
for i in range(1, 5):
    fig.add_subplot(2, 2, i)
    plt.subplot(221)
    values1, bins1, bars1 = plt.hist(reviews1['desert_snow'], edgecolor='white', bins=50, color='skyblue', range=(-1, 1))
    hist1, _ = np.histogram(reviews1['desert_snow'], bins=50, density=True)
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('desert&snow cosin(-1,1)')
    plt.subplot(223)
    values2, bins2, bars2 = plt.hist(reviews3['z_desert_snow'], edgecolor='white', bins=50, color='skyblue', range=(-1, 1))
    hist3, _ = np.histogram(reviews3['z_desert_snow'], bins=50, density=True)
    # 计算两个分布之间的KL散度
    kl_divergence1 = entropy(hist1, hist3)
    plt.text(x=-1, y=150, s='KL1:' + str(kl_divergence), fontdict=dict(fontsize=12, color='r', family='monospace', weight='normal'))
    plt.xlabel("cosin simi(-1,1)")
    plt.ylabel("Frequency")
    plt.title('z_desert&snow cosin(-1,1)')
```

![image-20230518172602436](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230518172602436.png)

1. 计算MMD
2. 圆形
3. soccer
