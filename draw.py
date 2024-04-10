import pandas as pd
import matplotlib.pyplot as plt

# 读取保存的评价指标文件
metrics_df = pd.read_csv('evaluation_metrics.csv')

# 绘制折线图
plt.figure(figsize=(10, 8))
plt.plot(metrics_df['Accuracy'], label='Accuracy')
plt.plot(metrics_df['F1 Score'], label='F1 Score')
plt.plot(metrics_df['Precision'], label='Precision')
plt.plot(metrics_df['Recall'], label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics Over Time')
plt.legend()
plt.show()
