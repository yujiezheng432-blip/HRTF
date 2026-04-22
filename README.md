我正在整理代码文档，下面是我当时的思维导图，里面有详细的思维路径，同时也标注了关键程序的作用以及目标。后续我会慢慢修改我的文档代码，并持续更新文档。

<img width="7228" height="9484" alt="SONICOM数据预测HRTF_SONICOM数据预测HRTF" src="https://github.com/user-attachments/assets/4f1527fb-7d7f-4823-8b95-e0b98c90f75b" />
<img width="1087" height="463" alt="SONICOM数据预测HRTF_用耳朵3D模型去生成或预测HRTF" src="https://github.com/user-attachments/assets/fb1e59ac-6e55-4aec-b426-104ade1b0899" />

<img width="2439" height="12990" alt="SONICOM数据预测HRTF_1 数据清洗" src="https://github.com/user-attachments/assets/830ab66e-705d-42ab-a729-679f245e24f1" />

<img width="2719" height="2932" alt="SONICOM数据预测HRTF_2 设计模型输入：耳朵图形数据==输出：单方向HRTF" src="https://github.com/user-attachments/assets/cf09416d-fd31-4e2d-b03d-cab0f35e6e0c" />
<img width="2581" height="4410" alt="SONICOM数据预测HRTF_3 扩大实验范围为全数据·300+" src="https://github.com/user-attachments/assets/41c8d0e2-40c0-444f-b846-724145071cc1" />

<img width="3006" height="9622" alt="SONICOM数据预测HRTF_4 训练" src="https://github.com/user-attachments/assets/a60dd85f-a248-4c80-ae53-3e2f7d7d8ca2" />
<img width="2502" height="7200" alt="SONICOM数据预测HRTF_6 3_TrainFromHRTFB3 py第三轮模型训练，保存为Basic3=============增加每个受试者-平均曲线，获得特异性部分修改模型不使用预训练模型（5 系列都不是使用的预训练参数，所以这一步省略）划分不参与训练的测试集" src="https://github.com/user-attachments/assets/f7573d52-c1e2-47d2-bfc0-4d407a259b10" />
<img width="1150" height="2124" alt="SONICOM数据预测HRTF_7==========增加点云的特征向量，从最开始的2048提升到16384，提升8倍。（因为原始点云数据平均约有7M个点，仅保留2048个点作为原始输入，会有很多local feature被丢失）" src="https://github.com/user-attachments/assets/8708525d-279e-4be5-a7e3-db487cd9c5e9" />
<img width="1405" height="1926" alt="SONICOM数据预测HRTF_8=========增加：1  将pointNet改成PointNet++。因为PointNet++可以学习到更多的局部特征，而PointNet只能学到全局特征。2  将头部切分出来，变成单独的数据集，datasetEarSTL，只采用头部甚至是耳廓数据进行训练" src="https://github.com/user-attachments/assets/9761723c-217d-4e33-84f7-bef0683926c4" />
<img width="996" height="1101" alt="SONICOM数据预测HRTF_9====全部改变，先建立最简单的端到端pipline。前端输入头部STL和对应方向sofa文件，后端得到全方位，双耳，frequency bins 第一步工作：将每个受试者的双耳HRTF数据拼接一起，得到单一受试者的全方位HRTF。同时保存每个HRTF的frequency bins以便后续分割不同方向上的hrtf第二步工作：构建原始pipline" src="https://github.com/user-attachments/assets/3a6a4939-510d-4d59-92ce-d23a7776ae15" />
