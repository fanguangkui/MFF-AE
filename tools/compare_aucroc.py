import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
font_options = [
    'Noto Sans CJK SC',
    'WenQuanYi Zen Hei',
    'Microsoft YaHei',
    'WenQuanYi Micro Hei'
]
plt.rcParams['font.sans-serif'] = font_options # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300  # 设置全局默认DPI:ml-citation{ref="7,9" data="citationList"}
plt.rcParams['font.size'] = 18  # 设置全局基础字体大小

colors = [
    '#74d2e7', '#8db9ca', '#c1d82f', '#00a4e4', '#8a7967',
    '#1cc7d0', '#2dde98', '#ffc168', '#ff6c5f', '#ff4f81',
    '#b84592', '#1769ff', '#fbb034', '#ffdd00', '#009f4d',
    '#ffc845', '#ffc20e',
]
def plot_compare_aucroc(x, aucroc_list, model_list):
    auc_roc_data = {
        'feature_shuffle_ratio': x
    }
    fig = plt.figure()
    for (idx, imodel) in enumerate(model_list):
        y = aucroc_list[imodel]['avg_aucroc']
        auc_roc_data[f"{imodel}_mean"] = y
        auc_roc_data[f"{imodel}_std"] = aucroc_list[imodel]['std_aucroc']

        # plt.errorbar(x, y, yerr=yerr, label=imodel)
        if imodel == 'HelaV1_SAE':
            model_name = "AE"
            color = "#3399cc"
        elif imodel == "HelaV1_AE-MEM10-S20":
            model_name = "MCOD (Ours)"
            color = "#123962"
        else:
            model_name = imodel
            color = colors[idx % len(colors)]
        print(imodel, color)

        model_name=model_name.replace("HelaV1_", "")
        plt.plot(x, y, label=model_name, color=color)
    plt.ylim([0, 1.1])
    plt.xlabel('Feature shuffle ratio')
    if draw_key == 'aucroc':
        plt.ylabel('Avg. AUC-ROC')
        pd.DataFrame(auc_roc_data).to_csv("auc_roc_data.csv")
    elif draw_key == 'aucpr':
        plt.ylabel('Avg. AUC-PR')
        pd.DataFrame(auc_roc_data).to_csv("auc_pr_data.csv")


    plt.legend(
        # ncol=2,
        loc='lower right', prop={'size': 12})
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"./avg_{draw_key}.png")

def plot_bar_aucroc(model_list, aucroc_list):
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7))
    mean_value = [_['avg_aucroc'][0] for m_name, _ in aucroc_list.items()]
    std_value = [_['std_aucroc'][0] for m_name, _ in aucroc_list.items()]
    model_namelist = [_.replace("HelaV1_AE-", "").replace("HelaV1_SAE", "Baseline").replace("MEM", "Win") for _ in model_list]
    # 绘制柱状图
    bars = ax.bar(model_namelist, mean_value, yerr=std_value, capsize=5,
                  color=colors,
                  alpha=0.8, edgecolor='black', linewidth=0.5)

    # 设置标题和标签
    if draw_key == 'aucroc':
        ax.set_ylabel('Avg. AUC-ROC')
        # 设置y轴范围
        ax.set_ylim(0.0, 0.8)
        offset = 0.05
    elif draw_key == 'aucpr':
        ax.set_ylabel(f'Avg. AUC-PR')
        # 设置y轴范围
        ax.set_ylim(0.0, 0.5)
        offset = 0.04


    # 在每个柱子上方添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f'{mean_value[i]:.3f}', ha='center', va='bottom', fontweight='bold',
                rotation=90
                )

    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加背景色
    ax.set_facecolor('#F5F5F5')

    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=15)

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

def plot_2bar_aucroc(aucroc_list):
    model_list1 = [
        'HelaV1_AE-MEM10-S20',
        'HelaV1_AE-MEM20-S20',
        'HelaV1_AE-MEM30-S20',
        'HelaV1_AE-MEM40-S20',
        'HelaV1_AE-MEM50-S20',
        'HelaV1_AE-MEM60-S20',
        'HelaV1_AE-MEM70-S20',
        'HelaV1_AE-MEM80-S20',
        'HelaV1_AE-MEM90-S20',
    ]

    model_list2 = [
        'HelaV1_AE-MEM10-S20',
        'HelaV1_AE-MEM20-S30',
        'HelaV1_AE-MEM30-S40',
        'HelaV1_AE-MEM40-S50',
        'HelaV1_AE-MEM50-S60',
        'HelaV1_AE-MEM60-S70',
        'HelaV1_AE-MEM70-S80',
        'HelaV1_AE-MEM80-S90',
        'HelaV1_AE-MEM90-S100',
    ]
    model_namelist1 = [
        _.replace("HelaV1_AE-MEM", "Win").split("-")[0] for _ in model_list1
    ]

    mean_value1 = [round(_['avg_aucroc'][0], 3) for m_name, _ in aucroc_list.items() if m_name in model_list1]
    std_value1 = [_['std_aucroc'][0] for m_name, _ in aucroc_list.items() if m_name in model_list1]

    mean_value2 = [round(_['avg_aucroc'][0], 3) for m_name, _ in aucroc_list.items() if m_name in model_list2]
    std_value2 = [_['std_aucroc'][0] for m_name, _ in aucroc_list.items() if m_name in model_list2]

    measurement_mean = [mean_value1, mean_value2]
    measurement_std = [std_value1, std_value2]
    species = ['Dynamic Window', "Fixed Window"]
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(model_namelist1))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    # 绘制柱状图
    # 设置标题和标签
    if draw_key == 'aucroc':
        txt_offset = 0.06
    elif draw_key == 'aucpr':
        txt_offset = 0.04

    colors = ["#ffb6b4", "#fcd8b5"]
    for idx in range(2):
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurement_mean[idx], width,
            yerr=measurement_std[idx], label=species[idx],
            capsize=5,
            # alpha=0.6,
            edgecolor='black',
            color=colors[idx],
            linewidth=0.5
        )
        # ax.bar_label(rects, padding=4, rotation=90)
        # 在每个柱子上方添加数值标签
        for i, bar in enumerate(rects):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + txt_offset,
                    f'{measurement_mean[idx][i]:.3f}',
                    ha='center', va='bottom',
                    fontweight='bold', rotation=90
                    )


        multiplier += 1

        # 设置标题和标签
        if draw_key == 'aucroc':
            ax.set_ylabel('Avg. AUC-ROC')
            # 设置y轴范围
            ax.set_ylim(0.0, 1.1)
            offset = 0.05
        elif draw_key == 'aucpr':
            ax.set_ylabel(f'Avg. AUC-PR')
            # 设置y轴范围
            ax.set_ylim(0.0, 0.5)
            offset = 0.04

    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加背景色
    ax.set_facecolor('#F5F5F5')

    ax.set_xticks(x + width / 2., model_namelist1, rotation=15)
    ax.legend(loc='upper left', ncols=3)

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == '__main__':
    #============== 1. get model list =======================#
    model_list = [
        'HelaV1_SAE',
        'HelaV1_DIF',
        'HelaV1_DeepSVDD',
        'HelaV1_SOGAAL',
        'HelaV1_VAE',

        'HelaV1_AE-ResLoss-CLS',
        'HelaV1_AE-ResLoss',
        'HelaV1_AE-CLS',

        'HelaV1_GMM',
        'HelaV1_KNN',
        'HelaV1_CBLOF',
        'HelaV1_COF',
        'HelaV1_FeatureBagging',
        'HelaV1_LSCP',
        'HelaV1_LOF',
        'HelaV1_ABOD',
        'HelaV1_ECOD',
        'HelaV1_HBOS',
        'HelaV1_IForest',
        'HelaV1_LMDD',
         'HelaV1_OCSVM',
        'HelaV1_PCA',

        # 'HelaV1_AE-CLS',
        # 'HelaV1_AE-MEM-NoOverDect-S20',
        # 'HelaV1_AE-MEM-NoOverDect-S80',
        # 'HelaV1_AE-MEM-NoOverDect-S20-CLS',
        # 'HelaV1_AE-MEM10-S80',
        # 'HelaV1_AE-MEM10-S20-CLS',

        # 'HelaV1_AE-MEM10-S20',
        # 'HelaV1_AE-MEM10-S30',
        # 'HelaV1_AE-MEM10-S40',
        # 'HelaV1_AE-MEM10-S50',
        # 'HelaV1_AE-MEM10-S60',
        # 'HelaV1_AE-MEM10-S70',
        # 'HelaV1_AE-MEM10-S80',
        # 'HelaV1_AE-MEM10-S90',
        # 'HelaV1_AE-MEM10-S100',
        #
        # 'HelaV1_AE-MEM10-S20',
        # 'HelaV1_AE-MEM20-S20',
        # 'HelaV1_AE-MEM30-S20',
        # 'HelaV1_AE-MEM40-S20',
        # 'HelaV1_AE-MEM50-S20',
        # 'HelaV1_AE-MEM60-S20',
        # 'HelaV1_AE-MEM70-S20',
        # 'HelaV1_AE-MEM80-S20',
        # 'HelaV1_AE-MEM90-S20',
        #
        # 'HelaV1_AE-MEM10-S20',
        # 'HelaV1_AE-MEM20-S30',
        # 'HelaV1_AE-MEM30-S40',
        # 'HelaV1_AE-MEM40-S50',
        # 'HelaV1_AE-MEM50-S60',
        # 'HelaV1_AE-MEM60-S70',
        # 'HelaV1_AE-MEM70-S80',
        # 'HelaV1_AE-MEM80-S90',
        # 'HelaV1_AE-MEM90-S100',

    ]

    # draw_key = 'aucpr'
    draw_key = 'aucroc'
    repeat_times = 10
    # feature_shuffle_ratio = [_ / 100.0 for _ in range(1, 41)]
    # res_floder = "Result_TestonALL"
    feature_shuffle_ratio = [_/1000.0 for _ in range(1, 21)]
    res_floder = "Result"
    # ============== 2. get avg and std aucroc of each model =======================#
    aucroc_list = {_ : {} for _ in model_list}
    for model_folder in model_list:
        for iratio in feature_shuffle_ratio:
            if f"ration_{iratio}" not in aucroc_list[model_folder]:
                aucroc_list[model_folder][f"ration_{iratio}"] = []
                aucroc_list[model_folder][f"avg_aucroc"] = []
                aucroc_list[model_folder][f"std_aucroc"] = []

            for irepeat in range(repeat_times):
                result_path = os.path.join(f'../{res_floder}/{model_folder}/Inference/fake-shuffle_ratio-{iratio}-repeat_time-{irepeat}', 'results.json')
                with open(result_path, 'r') as fcc_file:
                    fcc_data = json.load(fcc_file)
                aucroc_list[model_folder][f"ration_{iratio}"].append(fcc_data[draw_key])
    for model_folder in model_list:
        aucroc_info = aucroc_list[model_folder]
        for iratio in feature_shuffle_ratio:
            avg_aucroc = np.mean(aucroc_info[f"ration_{iratio}"])
            std_aucroc = np.std(aucroc_info[f"ration_{iratio}"], ddof=0)
            aucroc_list[model_folder][f"avg_aucroc"].append(avg_aucroc)
            aucroc_list[model_folder][f"std_aucroc"].append(std_aucroc)
    print(aucroc_list)
    # ============== 3. plot trend line of ratio of each model =======================#


    plot_compare_aucroc(feature_shuffle_ratio, aucroc_list, model_list)
    # plot_bar_aucroc(model_list, aucroc_list)
    # plot_2bar_aucroc(aucroc_list)