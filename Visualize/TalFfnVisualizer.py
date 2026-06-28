#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAL-FFN实验结果可视化模块
用于生成论文所需的各类图表 (正式版)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

class TALFFNVisualizer:
    """
    TAL-FFN实验结果可视化类
    """

    def __init__(self, output_dir='./results/figures', style='paper'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style

        # 设置中文字体与SCI样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False

        if style == 'paper':
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['savefig.dpi'] = 300
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.labelsize'] = 11
            plt.rcParams['axes.titlesize'] = 12
        
    def plot_ablation_stages(self, results_data, save_path=None):
        """消融实验对比柱状图与曲线图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: mAP对比
        ax1 = axes[0]
        x = np.arange(len(results_data))
        width = 0.6
        
        bars = ax1.bar(x, results_data['mAP50'], width, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        ax1.set_ylabel('mAP@50 (%)', fontweight='bold')
        ax1.set_title('(a) Detection Accuracy', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_data['stage'], rotation=15)
        ax1.set_ylim(98.0, 99.2)
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # 右图: 参数量趋势
        ax2 = axes[1]
        ax2.plot(x, results_data['params'], marker='o', markersize=8, color='#e74c3c', linewidth=2, label='Parameters (M)')
        
        ax2.set_ylabel('Parameters (M)', fontweight='bold', color='#e74c3c')
        ax2.set_title('(b) Model Complexity', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_data['stage'], rotation=15)
        ax2.grid(linestyle='--', alpha=0.6)
        
        for i, val in enumerate(results_data['params']):
            ax2.text(i, val + 0.1, f'{val:.2f}', ha='center', color='#c0392b', fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        return fig

    def plot_sota_comparison(self, sota_data, save_path=None):
        """SOTA 对比雷达图或综合对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = sota_data['Model']
        map50 = sota_data['mAP50']
        params = sota_data['Params']
        
        # 归一化处理以便在同图中对比
        norm_map = (map50 - 97.0) / (99.5 - 97.0) 
        norm_params = 1.0 - (params / params.max()) # 参数越小越好
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, norm_map, width, label='Normalized Accuracy (mAP50)', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, norm_params, width, label='Normalized Complexity (1/Params)', color='#9b59b6', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold')
        ax.set_title('SOTA Comparison (Trade-off Analysis)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"SOTA plot saved to {save_path}")
        return fig

if __name__ == '__main__':
    # 1. 消融实验数据 (Table 1)
    ablation_df = pd.DataFrame({
        'stage': ['Baseline', '+P2 Layer', '+TAL-FFN', 'AgriYOLO'],
        'mAP50': [98.7, 98.8, 98.8, 98.9],
        'params': [8.09, 6.50, 6.42, 6.24]
    })
    
    # 2. SOTA 对比数据 (Table 2)
    sota_df = pd.DataFrame({
        'Model': ['YOLOv5s', 'YOLOv8s', 'YOLOv9c', 'YOLOv10s', 'AgriYOLO'],
        'mAP50': [98.29, 97.40, 98.19, 98.73, 98.90],
        'Params': [2.65, 3.16, 25.59, 8.09, 6.24],
        'FPS': [163.9, 149.2, 60.0, 82.4, 71.7]
    })

    viz = TALFFNVisualizer(output_dir='e:/Desktop/yolov10-main/results/figures')
    viz.plot_ablation_stages(ablation_df, save_path='e:/Desktop/yolov10-main/results/figures/fig_ablation_sci.png')
    viz.plot_sota_comparison(sota_df, save_path='e:/Desktop/yolov10-main/results/figures/fig_sota_sci.png')
