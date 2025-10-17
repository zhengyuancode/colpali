from pymilvus import MilvusClient, DataType
import json
import os

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

def delete_collection(collection_name):
    return client.drop_collection(collection_name=collection_name)

print(client.list_collections())
# print(delete_collection("MMLongDoc_text"))

# from transformers import pipeline

# pipe = pipeline("image-text-to-text", model="XiaomiMiMo/MiMo-VL-7B-RL-2508")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     },
# ]
# pipe(text=messages)


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import make_interp_spline

# # Updated data (text block screening only)
# topk_values = [10, 15, 20, 30]  # Added topk*50 for convergence
# time_values = [1.04, 1.24, 1.45, 2.21]  # Time approaching 4.2h
# acc_values = [79.4, 79.7, 80.2, 80.8]  # Accuracy approaching 81.6%

# # Feature point screening result (topk*10) to mark with horizontal lines
# feature_time = 4.2
# feature_acc = 81.6

# # Generate smooth curves with extended data
# x_smooth = np.linspace(min(topk_values), max(topk_values), 300)
# time_spline = make_interp_spline(topk_values, time_values)
# acc_spline = make_interp_spline(topk_values, acc_values)
# time_smooth = time_spline(x_smooth)
# acc_smooth = acc_spline(x_smooth)

# # Create figure and dual y-axes
# fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

# # Plot smooth time curve (BLUE)
# color_time = 'tab:blue'
# ax1.set_xlabel('Topk Multiplier', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Time (h)', color=color_time, fontsize=12, fontweight='bold')
# ax1.plot(x_smooth, time_smooth, color=color_time, linewidth=2.5, alpha=0.8, 
#          label='Text Block Time', zorder=2)
# ax1.tick_params(axis='y', labelcolor=color_time)
# ax1.grid(True, linestyle='--', alpha=0.6)

# # Create second y-axis for accuracy (RED) - with adjusted scale
# ax2 = ax1.twinx()
# color_acc = 'tab:red'
# ax2.set_ylabel('Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
# ax2.plot(x_smooth, acc_smooth, color=color_acc, linewidth=2.5, linestyle='--', 
#          alpha=0.8, label='Text Block Accuracy', zorder=2)
# ax2.tick_params(axis='y', labelcolor=color_acc)

# # Add horizontal lines for feature point screening (time and accuracy)
# ax1.axhline(y=feature_time, color=color_time, linestyle='--', alpha=0.6, 
#             label=f'Feature Point Time ({feature_time}h)')
# ax2.axhline(y=feature_acc, color=color_acc, linestyle='--', alpha=0.6, 
#             label=f'Feature Point Accuracy ({feature_acc}%)')

# # Adjust accuracy axis range to 50-100 as requested
# ax1.set_xlim(8, 32)
# ax1.set_ylim(0, 6)
# ax2.set_ylim(60, 100)  # THIS IS THE KEY CHANGE - Set to 50-100

# # Add convergence annotation
# ax1.annotate("Convergence toward feature point", 
#              xy=(45, 3.9),
#              xytext=(45, 3.9),
#              fontsize=10, color='darkgray', alpha=0.7)

# # Add title and legend
# plt.title('Impact of Topk Multiplier on Time and Accuracy', 
#           fontsize=14, fontweight='bold', pad=15)
# fig.tight_layout()

# # Create custom legend with distinct colors
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
#            fontsize=10, frameon=True, framealpha=0.9)

# # Save high-resolution figure for publication
# plt.savefig('convergence_trend.png', bbox_inches='tight', dpi=300)
# plt.show()