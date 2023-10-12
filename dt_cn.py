import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import local_packages.deformationTransfer as dt
import local_packages.tools3d_ as t3d

import landmarks.LICT_narrow_r as LICT_narrow
import landmarks.LARkit as LARkit

# 目标面路径
path_target_face = 'data/Neutral.obj'
# NRICP路径
path_NRICP = 'data/ictinner2ARkit.obj'
# 源混合形状目录
path_in = 'data/ARKit_blendShapes/'
# 输出目录
path_out = 'dt_results/'

# 从源混合形状和目标面读取地标点
target_lm = LICT_narrow.LM[0:9]
source_lm = LARkit.LM[0:9]

# 读取目标面（原始+NRICP）和源中性面
source_vertices, source_faces, source_quads, _ = t3d.Read(path_in + 'Neutral.obj', QuadMode=True)
target_vertices, target_faces, target_quads, _ = t3d.Read(path_target_face, QuadMode=True)
# 使用Wrap3D创建的
deformed_target_vertices, _, _, _ = t3d.Read(path_NRICP, QuadMode=True)

print("target_vertices->", target_vertices.shape)
print("target_faces->", target_faces.shape)
t3d.ShowMesh(target_vertices, target_faces)
print("deformed_vertices->", deformed_target_vertices.shape)
t3d.ShowMesh(deformed_target_vertices, target_faces)
print("\nsource_vertices->", source_vertices.shape)  # 检查源和目标拓扑的形状
print("source_faces->", source_faces.shape)
t3d.ShowMesh(source_vertices, source_faces)

# 刚性对齐目标面到源面，以找到良好的对应关系并在变形传递之前达到正确的比例
# 可视化检查NRICP匹配是否良好
target_lm = LICT_narrow.LM[0:5]
source_lm = LARkit.LM[0:5]

# 将变形的目标面（NRICP）与源面网格对齐以找到良好的对应关系
print("Alignment of deformed target face")
deformed_target_vertices = t3d.align_target_to_source(
    deformed_target_vertices, target_faces, target_lm, source_vertices, source_faces, source_lm
)
t3d.Show2Meshes(deformed_target_vertices, target_faces, source_vertices, source_faces)
# 对齐原始目标面以进行良好的变形传递（与源面网格具有相同的比例和方向）
print("Alignment of original target face")
target_vertices = t3d.align_target_to_source(
    target_vertices, target_faces, target_lm, source_vertices, source_faces, source_lm
)
t3d.Show2Meshes(target_vertices, target_faces, source_vertices, source_faces)

# 将数据转换为[:,3]格式
source_vertices = source_vertices.T
source_faces = source_faces.T
target_vertices = target_vertices.T
target_faces = target_faces.T
deformed_target_vertices = deformed_target_vertices.T

start_time_1 = time.time()

print("Compute source_v4, target_v4 and taget_V_inverse...")
start_time = time.time()
target_v4 = dt.compute_v4(target_vertices, target_faces)
source_v4 = dt.compute_v4(source_vertices, source_faces)
target_V_inverse = dt.compute_V_inverse(target_vertices, target_faces, target_v4)
print("done in", (time.time() - start_time), "sec")

print("Generating matrices...")
# 变形平滑度，ES，表示相邻三角形的变换应该相等
start_time = time.time()
Es_ATA, Es_ATc = dt.makeEs_ATA_ATc(target_vertices, target_faces, target_V_inverse)
print("Es :", (time.time() - start_time), "sec")

# 变形恒等性，EI，当所有变换都等于单位矩阵时最小化：
start_time = time.time()
Ei_ATA, Ei_ATc = dt.makeEi_ATA_ATc(target_vertices, target_faces, target_V_inverse)
print("Ei :", (time.time() - start_time), "sec")

start_time = time.time()
correspondences = dt.get_correspondece_faces(
    source_vertices, source_faces, deformed_target_vertices, target_faces
)
print("\ndone in ", (time.time() - start_time), "sec")

print("Generating deformation transfer matrices...")
# 最接近有效点的项，Ed，表示源网格的每个顶点的位置应等于目标网格上最接近的有效点。
start_time = time.time()
Ed_A = dt.makeEd_A(correspondences, target_vertices, target_faces, target_V_inverse)
Ed_ATA = np.dot(Ed_A.T, Ed_A)
elapsed_time = time.time() - start_time
print("Ed_A, Ed_ATA :", elapsed_time, "sec")

elapsed_time = time.time() - start_time_1
print("\nOne-off computation finished in", elapsed_time, "sec\n\n")

##################### 上述计算仅计算一次。
##################### 对于每个要传递的新变形，我们只从这里开始计算。
########################## 批处理过程 ########################################

start_time_2 = time.time()
print('\nBatch process ')

source_data = os.scandir(path_in)
n_data = len(
    [blend_shape for blend_shape in os.listdir(path_in) if os.path.splitext(blend_shape)[1] == ".obj"]
)  # 要处理的数据点数
print("Applying Deformation Transfer to ", n_data, "blend shapes...\n")
start_time_all = time.time()

for blend_shape in source_data:
    name, ext = os.path.splitext(blend_shape)
    name = name.split("/")
    if ext == ".obj":  # 仅读取源目录中的.obj文件
        print("\nworking on", blend_shape.name)
        objpath = path_in + blend_shape.name
        source_vertices2, _, _, _ = t3d.Read(objpath, QuadMode=True)

        # 变形传递前的对齐（不适用 - 在预处理中已经处理）
        source_vertices2 = source_vertices2.T
        # source_vertices2 = t3dtools_3d.align_target_to_source(source_vertices2.T, source_faces.T, skull_landmaks_source, source_vertices.T, source_faces.T, skull_landmaks_source).T

        # 计算新的源旋转矩阵
        source_rotation = dt.make_source_rotation_matrix(
            source_vertices, source_faces, source_v4, source_vertices2, source_faces
        )

        # 回代步骤
        print("Make Ed_ATc...   ")
        start_time = time.time()
        # 最接近有效点的项，Ed，表示源网格的每个顶点的位置应等于目标网格上最接近的有效点。
        start_time = time.time()
        Ed_ATc = dt.makeEd_ATc(correspondences, source_rotation, Ed_A)
        elapsed_time = time.time() - start_time
        print("done in ", elapsed_time, "sec")

        start_time_solution = time.time()
        print("Solving Matrix system...")
        wd = 1
        wi = 0.01
        ws = 0.01  # 标准值：wd=1; wi=0.001; ws=0.01; 我们选择非常低的恒等权重以准确传递变形
        # 这导致一些全局不需要的变形，我们在后处理中修复

        ATA_sum = wd * Ed_ATA + wi * Ei_ATA + ws * Es_ATA
        ATc_sum = wd * Ed_ATc + wi * Ei_ATc + ws * Es_ATc

        x = spsolve(ATA_sum, ATc_sum)

        elapsed_time = time.time() - start_time_solution
        print("\n calculation was finished in", elapsed_time, "sec")
        target_vertices2 = x[0 : len(target_vertices) * 3].reshape(len(target_vertices), 3)

        # 保存
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        t3d.SaveObj(
            target_vertices2.T,
            target_faces.T,
            path_target_face,
            save_destination=path_out + name[-1] + ".obj",
            CM=True,
        )

elapsed_time = time.time() - start_time_2
print("\n\n Batch-process calculations finished in", elapsed_time, "sec")
