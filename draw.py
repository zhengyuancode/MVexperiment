from graphviz import Digraph

# 创建流程图
dot = Digraph(comment='实验流程框图', node_attr={'fontname': 'SimHei'})
dot.attr(rankdir='TB', size='12,8', dpi='300')


dot.node('data', '数据准备\n1. 加载D4LA\n2. 图像预处理\n3. DataLoader', 
         shape='oval', pos='300,800!')  

dot.node('baseline', '基线模型\nUNet', 
         shape='oval', pos='200,600!')  
dot.node('attention', '注意力模型\nUNet_Attention', 
         shape='oval', pos='400,600!')  


dot.edge('data', 'baseline', '→', headport='n', tailport='s')
dot.edge('data', 'attention', '→', headport='n', tailport='s')


dot.node('train', '模型训练\n1. 前向传播\n2. 损失优化\n3. 保存权重', 
         shape='oval', pos='300,400!')
dot.edge('baseline', 'train', '→', headport='n', tailport='s')
dot.edge('attention', 'train', '→', headport='n', tailport='s')


dot.node('eval', '模型评估\n1. 测试集推理\n2. 计算mIoU/Acc\n3. 计算Dice系数\n4. 计算像素精度(PA)', 
         shape='oval', pos='300,200!')
dot.edge('train', 'eval', '→', headport='n', tailport='s')

dot.node('vis', '结果可视化\n1. 预测对比\n2. 指标表格', 
         shape='oval', pos='300,0!')
dot.edge('eval', 'vis', '→', headport='n', tailport='s')

dot.render('experiment_block_diagram', format='png', cleanup=True, view=True)