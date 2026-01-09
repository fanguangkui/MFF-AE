import numpy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
from DetectOutlier.utils.parameters import prefix_str2para
import matplotlib
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
# from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler

font_options = [
    'Noto Sans CJK SC',
    'WenQuanYi Zen Hei',
    'Microsoft YaHei',
    'WenQuanYi Micro Hei'
]
plt.rcParams['font.sans-serif'] = font_options # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300  # 设置全局默认DPI:ml-citation{ref="7,9" data="citationList"}


clist=['#4169E1','#87CEEB']
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 18

def anno_scatter(x, y, label, shape, save_path):
    fig, ax = plt.subplots()

    for xi, yi, ilabel, ishape in zip(x, y, label, shape):
        plt.scatter(xi, yi, label=ilabel, marker=ishape)

        # produce a legend with the unique colors from the scatter

    ax.legend(
        loc="best", title="Model Name"
    )
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Confidence')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)


# ====================================================== #
def visualize_all_single_bar(pseudo_data, save_path):
    fig, axs = plt.subplots(nrows=3, ncols=3,
                           figsize=(15, 15))
    model_list = pseudo_data["model_name"].unique().tolist()
    model_list.append("AvgKNN")
    outlier_list = pseudo_data["sample_name"].unique().tolist()

    outlier_info = {_: {
        "outlier_list": outlier_list,
        "outlier_num": [],
        "inlier_num": [],
    } for _ in model_list}

    for idx, model_name in enumerate(model_list):
        if model_name == "AvgKNN":
            bool_index = pseudo_data.hyper_list.str.contains('method-mean')
            model_data = pseudo_data[bool_index]
        elif model_name == "KNN":
            bool_index = pseudo_data.hyper_list.str.contains('method-largest')
            model_data = pseudo_data[bool_index]
        else:
            model_data = pseudo_data[pseudo_data['model_name']==model_name]
        for ioutlier in outlier_list:
            model_sample_data = model_data[model_data['sample_name']==ioutlier]
            outlier_num = model_sample_data[model_sample_data['classify']=="outlier"].shape[0]
            inlier_num = model_sample_data[model_sample_data['classify']=="inlier"].shape[0]
            outlier_info[model_name]['outlier_num'].append(outlier_num)
            outlier_info[model_name]['inlier_num'].append(inlier_num)


    width = 0.7

    for idx, (model_name, model_info) in enumerate(outlier_info.items()):
        axs[idx//3][idx%3].set_title(model_name)

        labels = model_info['outlier_list']
        inliners = model_info['inlier_num']
        outliners = model_info['outlier_num']
        axs[idx//3][idx%3].bar(labels, outliners, width, label='Outlier', color="#FF5F6B")
        axs[idx//3][idx%3].bar(labels, inliners, width, bottom=outliners, label='Non-outlier', color="#808080")

        # axs[idx//3][idx%3].set_yticklabels(labels=range(20))
        axs[idx//3][idx%3].set_xticklabels(labels=labels, rotation=60, ha='right')
        axs[idx//3][idx%3].set_ylim(0, 19)
        axs[idx//3][idx%3].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # 绘制data completeness
    # axs[2][1].set_title("data completeness")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
# ====================================================== #
def visualize_num_bar(anomly_list_info, save_path, mark_list_info=()):
    # [
    #     {
    #         'name': ip_name,
    #         'outlier_num': sample_df['outlier'].sum(),
    #         'normal_num': 0,
    #         'p_value': 0,
    #     },...
    # ]

    marked_species = []
    # marked_p_value = []
    for _ in mark_list_info:
        marked_species.append(_['name'])
        # marked_p_value.append(round(_['p_value'], 3))

    species = []
    anomaly = []
    normality = []
    score_value = []
    for _ in anomly_list_info:
        species.append(_['name'])
        anomaly.append(_['outlier_num'])
        normality.append(_['normal_num'])
        if _['name'] in marked_species:
            score_value.append("*")
        else:
            score_value.append("")

    fig = plt.figure()
    ax = fig.gca()
    bottom = np.zeros(len(species))

    state_counts = {
        'anomaly counts': np.array(anomaly),
        'inlier counts': np.array(normality),
    }
    print(state_counts)
    bar_heightest = 0
    for state, state_count in state_counts.items():
        if "inlier" in state:
            color ="#808080"
        else:
            color = "#FF5F6B"
        bar_plot = ax.bar(species, state_count, label=state, bottom=bottom, color=color)
        bottom += state_count

    for idx, rect in enumerate(bar_plot):
        if bottom[idx] > bar_heightest:
            bar_heightest = bottom[idx]
        ax.text(rect.get_x() + rect.get_width() / 2., 0.5 + bottom[idx],
                score_value[idx],
                ha='center', va='bottom', rotation=0)
        # ax.bar_label(p, label_type='center')
    bottom, top = plt.ylim()
    plt.ylim(bottom, top + bar_heightest / 6)
    ax.legend(loc='lower right')
    plt.xticks(rotation=45)
    ax.set_xlabel('Sample Name')
    ax.set_ylabel('Count of determinations')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

# ====================================================== #
def plot_model_acc_trend(model_type_list, result_info, iindex, save_path):
    fig, axs = plt.subplots(
        nrows=5, ncols=2, sharex=True, sharey=True,
        figsize=(12, 24)
    )
    for model_type_id, model_type in enumerate(model_type_list):
        axs[model_type_id//2][model_type_id%2].set_title(model_type_list[model_type_id])

    for idx, (model_name_para, acc_info) in enumerate(result_info.items()):
        para = prefix_str2para(model_name_para)
        if para['model_name'] == "KNN":
            if "method-mean" in model_name_para:
                model_type_id = model_type_list.index("AvgKNN")
            else:
                model_type_id = model_type_list.index("KNN")
        else:
            model_type_id = model_type_list.index(para['model_name'])

        sd_acc = acc_info["sd_acc"]
        cv_acc = acc_info["cv_acc"]
        mean_acc = acc_info["mean_acc"]
        se_acc = acc_info["se_acc"]
        shuffle_ratio = acc_info["shuffle_ratio"]
        if iindex == 'mean_acc':
            axs[model_type_id//2][model_type_id%2].errorbar(
                range(len(shuffle_ratio)), mean_acc,
                yerr=se_acc, label=idx
            )
        elif iindex == 'sd_acc':
            axs[model_type_id//2][model_type_id%2].plot(
                range(len(shuffle_ratio)), sd_acc, label=idx
            )
        elif iindex == "cv_acc":
            axs[model_type_id//2][model_type_id%2].plot(
                range(len(shuffle_ratio)), cv_acc, label=idx
            )
        else:
            raise "Unknown index"
    # plt.show()
    # fig.suptitle(iindex)
    plt.xticks(range(len(shuffle_ratio)), shuffle_ratio)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

def plot_model_trend_pdf(model_type_list, result_info, iindex, save_path):
    # gridspec inside gridspec
    fig = plt.figure(figsize=(12, 12))
    gs0 = gridspec.GridSpec(3, 3, figure=fig)
    for i in range(8):
        fig.add_subplot(gs0[i])

    gs01 = gs0[8].subgridspec(2, 1)
    fig.add_subplot(gs01[0])
    fig.add_subplot(gs01[1])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    # if iindex == 'mean_acc':
    #     plt.suptitle("Mean Accuracy")
    # elif iindex == "cv_acc":
    #     plt.suptitle("CV of Accuracy")

    for idx, (model_name_para, acc_info) in enumerate(result_info.items()):
        para = prefix_str2para(model_name_para)
        if para['model_name'] == "KNN":
            if "method-mean" in model_name_para:
                model_type_id = model_type_list.index("AvgKNN")
            else:
                model_type_id = model_type_list.index("KNN")
        else:
            model_type_id = model_type_list.index(para['model_name'])

        shuffle_ratio = acc_info["shuffle_ratio"]
        if iindex == 'mean_acc':
            mean_acc = acc_info["mean_acc"]
            se_acc = acc_info["se_acc"]
            fig.axes[model_type_id].errorbar(
                range(len(shuffle_ratio)), [_*100 for _ in mean_acc],
                yerr=se_acc
            )
            label_x = 0.7
            label_y = 0.48
        elif iindex == "cv_acc":
            cv_acc = acc_info["cv_acc"]
            fig.axes[model_type_id].plot(
                range(len(shuffle_ratio)), cv_acc
            )
            # fig.axes[model_type_id].set_ylim(0, 1.4)
            label_x = 0.7
            label_y = 0.48
        # label_shuffle_ratio = []
        # for x in enumerate(shuffle_ratio):
        #     if x%2 == 0:
        #         label_shuffle_ratio.append(shuffle_ratio[x])
        fig.axes[model_type_id].set_xticks(
            range(len(shuffle_ratio)), [f"{str(int(_*100))}" for _ in shuffle_ratio],
            fontsize=12, rotation=90
        )

    for model_type_id, model_type in enumerate(model_type_list):
        # fig.axes[model_type_id].set_title(model_type_list[model_type_id])
        fig.axes[model_type_id].text(
            label_x, label_y, model_type_list[model_type_id].replace("FeatureBagging", "FeatB"),
            transform=fig.axes[model_type_id].transAxes,
            fontsize=14,
            verticalalignment='top', bbox=props
        )


        if model_type_id in [0, 3, 6]:
            if iindex == 'mean_acc':
                fig.axes[model_type_id].set_ylabel(f"Avg. Accuracy (%)", fontsize=14)
            elif iindex == "cv_acc":
                fig.axes[model_type_id].set_ylabel(f"Coefficient of Variation ", fontsize=14)
        if model_type_id in [6, 7, 9]:
            fig.axes[model_type_id].set_xlabel(f"Feature shuffle proportion (%)")
        # close x ticks
        if model_type_id in [0,1,2,3,4,5, 8]:
            fig.axes[model_type_id].set_xticks([])
        # close y ticks
        # if model_type_id in [1,2, 4, 5, 7]:
        #     fig.axes[model_type_id].set_yticks([])
    plt.tight_layout()
    # plt.legend()
    # plt.show()
    # print("aaaa")
    print(f"Trend is saved in {save_path}")
    plt.savefig(save_path, dpi=600)


def get_combination_result(
        data, model_list, outlier_scores,
        _outlier_integrated_id, save_path,
        gt_sample_name=()
):

    def contourf(outlier_scores, group_num=100):
        # colors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
        # cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        # mpl.colormaps.register(cmap=my_cmap)
        N = 100
        x = np.linspace(0., 1.1, N)
        y = np.linspace(1.1, 0, N)
        X, Y = np.meshgrid(x, y)
        z = np.zeros_like(X) #  z value
        xhist, xbin_edges = np.histogram(outlier_scores[model_name1], bins=group_num, range=(0.0, 1.0))
        yhist, ybin_edges = np.histogram(outlier_scores[model_name2], bins=group_num, range=(0.0, 1.0))
        for idx, (zx, zy) in enumerate(zip(np.cumsum(xhist), np.cumsum(yhist))):
            z[:, idx:idx+1] += zx
            z[group_num-idx-1:group_num-idx, :] += zy
        # RdOr = plt.cm.Blues_r.reversed()
        newcmp = LinearSegmentedColormap.from_list('chaos', clist)
        cs = ax.contourf(X, Y, z, rstride=1,cstride=1,cmap=newcmp)
        # cs = ax.contourf(X, Y, z, rstride=1,cstride=1,cmap=RdOr)

    model_name1, model_name2 = model_list
    data1 = data[model_name1]
    data2 = data[model_name2]
    outlier_id = []
    for _ in _outlier_integrated_id:
        # if _['name'] == 'iBAQ 17':
        #     continue
        outlier_id.append(_['name'])

    # outlier_id = [_['name'] for _ in _outlier_integrated_id]

    sorted_keys = sorted(data1)
    x_inlier_data = []
    y_inlier_data = []
    x_outlier_data = []
    y_outlier_data = []

    annotation_sample_name = []
    x_gt_data = []
    y_gt_data = []
    for ikey in sorted_keys:

        if ikey in outlier_id:
            annotation_sample_name.append(ikey)
            x_outlier_data.append(np.mean(data1[ikey]['outlier_score']))
            y_outlier_data.append(np.mean(data2[ikey]['outlier_score']))
            if ikey in gt_sample_name:
                x_gt_data.append(np.mean(data1[ikey]['outlier_score']))
                y_gt_data.append(np.mean(data2[ikey]['outlier_score']))
        else:
            x_inlier_data.append(np.mean(data1[ikey]['inlier_score']))
            y_inlier_data.append(np.mean(data2[ikey]['inlier_score']))
            if ikey in gt_sample_name:
                x_gt_data.append(np.mean(data1[ikey]['inlier_score']))
                y_gt_data.append(np.mean(data2[ikey]['inlier_score']))


    fig, ax = plt.subplots(figsize=(7, 7))
    contourf(outlier_scores)

    b = ax.scatter(
        x_inlier_data,
        y_inlier_data,
        c='white',s=35, marker="s", edgecolor='#808080'
    )
    c = ax.scatter(
        x_outlier_data,
        y_outlier_data,
        c='red', s=35, edgecolor='red'
    )

    d = ax.scatter(
        x_gt_data,
        y_gt_data,
        c='white', s=5, marker="s" ,edgecolor='k'
    )
    ax.axis('tight')
    if len(gt_sample_name)!=0:
        ax.legend(
            [b, c, d],
            ['Ensemble inliers', 'Ensemble outliers', 'Pseudo outliers'],
            prop=matplotlib.font_manager.FontProperties(size=15),
            loc='lower right')
    else:
        ax.legend(
            [b, c],
            ['Non-outlier', 'Outlier'],
            prop=matplotlib.font_manager.FontProperties(size=15),
            loc='lower right')
        # add labels to all points
        text_list = []
        for (xi, yi, sname) in zip(x_outlier_data, y_outlier_data, annotation_sample_name):
            text_list.append(plt.text(xi, yi, sname.replace("iBAQ ", "H")))
        adjust_text(text_list,)

    ax.set_xlabel(f"{model_name1} Decision Score", size = 21)
    ax.set_ylabel(f"{model_name2} Decision Score", size = 21)
    plt.yticks(fontproperties = 'Times New Roman', size = 21)
    plt.xticks(fontproperties = 'Times New Roman', size = 21)
    # ax.set_xlim((-0.01, 1.05))
    # ax.set_ylim((-0.01, 1.05))
    # plt.show()
    plt.tight_layout()
    # plt.show()
    # save_path = os.path.join(output_path, f'{}_correlation_map.jpg')
    np.save(
        os.path.join(os.path.split(save_path)[0], f'{model_name1}_{model_name2}_decision_boundary.npy'),
        {
            "x_inlier_data": x_inlier_data,
            "y_inlier_data": y_inlier_data,
            "x_outlier_data": x_outlier_data,
            "y_outlier_data": y_outlier_data,
        }
    )
    ax.figure.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"save plot results in {save_path}")
    plt.close()


def plot_gan_loss(train_history):
    """
    plot training history given the history dictionary.
    """
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    # aucy = train_history['auc']
    x = np.linspace(1, len(dy), len(dy))

    plt.plot(x, dy, color='blue', label='Discriminator Loss')
    plt.plot(x, gy, color='red', label='Generator Loss')
    # plt.plot(x, aucy, color='yellow', linewidth='3', label='ROC, AUC={}'.format(round(aucy[-1], 4)))

    plt.legend(loc='best')
    plt.show()

def visualize_error_feat(feat_dist, feat_label=None, save_name=''):
    rows, cols = 81, 81
    x = np.arange(rows)
    y = np.arange(cols)

    padded_arr = np.pad(feat_label, (0, rows*cols-len(feat_label)), mode='constant', constant_values=0)
    Z = padded_arr.reshape(cols, rows)  # 形状变为(80, 81)

    padded_feat_error = np.pad(feat_dist, (0, rows * cols - len(feat_dist)), mode='constant', constant_values=0)
    Z_feat_error = padded_feat_error.reshape(cols, rows)

    # 创建图形和子图
    fig = plt.figure(figsize=(8, 8))

    # 子图1: 3D 图
    ax1 = fig.add_subplot(111)
    # cmap = ListedColormap(['white', '#f0f0f0'])
    colors = [(1, 1, 1), (0.5, 0.5, 0.5)]  # 白色(1,1,1)到中灰色(0.5,0.5,0.5)

    cmap_white_gray = LinearSegmentedColormap.from_list('white_gray', colors)

    heatmap = ax1.imshow(Z_feat_error, cmap=cmap_white_gray, aspect='auto')  # 关键: 使用相同的范围
    # ax1.set_title('data map for original data', fontsize=14)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    fig.colorbar(heatmap, ax=ax1, shrink=0.6, label='feature` value')

    # # 子图2: 热力图
    # ax2 = fig.add_subplot(122)
    # heatmap = ax2.imshow(Z_feat_error, cmap='plasma', aspect='auto',
    #                      extent=[x.min(), x.max(), y.max(), y.min()])  # 关键: 使用相同的范围
    # ax2.set_xlabel('X', fontsize=12)
    # ax2.set_ylabel('Y', fontsize=12)
    # ax2.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    # # ax2.invert_yaxis()  # 反转 Y 轴以匹配 3D 图的方向
    # fig.colorbar(heatmap, ax=ax2, shrink=0.6, label='error value')


    if feat_label is not None:
        padded_feat_label = np.pad(feat_label, (0, rows * cols - len(feat_label)), mode='constant', constant_values=0)
        Z_feat_label = padded_feat_label.reshape(cols, rows)

        # 获取当前坐标轴的范围
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()

        # 计算每个单元格的中心坐标
        cell_width = (x_max - x_min) / Z_feat_label.shape[1]
        cell_height = (y_max - y_min) / Z_feat_label.shape[0]

        # 标记值为1的位置 - 精确计算中心坐标
        for i in range(Z_feat_label.shape[0]):
            for j in range(Z_feat_label.shape[1]):
                if Z_feat_label[i, j] == 1:
                    x_center = j
                    y_center = i-cell_height*0.35
                    # 左下到右上的斜线
                    ax1.text(x_center, y_center, '*',  # 可以使用 '*' 或 '★'
                            ha='center', va='center',
                            fontsize=8, color='red', fontweight='bold', transform=ax1.transData)
                    # ax2.text(j, i, '★',  # 可以使用 '*' 或 '★'
                    #          ha='center', va='center',
                    #          fontsize=5, color='black', fontweight='bold')

    # 确保两个子图的 X 轴范围相同
    ax1.set_xlim(x.min(), x.max())
    # ax2.set_xlim(x.min(), x.max())

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    # plt.show()
    plt.savefig(f"{save_name}.png")
    plt.close()

def visualize_tsne(feat, sample_label, save_name):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(feat)
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=sample_label.ravel(),  # 将标签展平为一维数组
        s=40  # 点的大小
    )
    # 添加图例和标签
    plt.colorbar(scatter, label='Label')
    plt.title('t-SNE Visualization of Data with 0/1 Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 显示图形
    plt.show()

def visualize_KDE(data, feature_name, labels, save_name):
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 分离两类数据
    class0_feature = data_scaled[labels == 0][:, 0]
    class1_feature = data_scaled[labels == 1][:, 0]
    recon_feature = data_scaled[labels == 0][:, 1]
    recon_abfeature = data_scaled[labels == 1][:, 1]

    # 创建可视化图形
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex='col', sharey='row')
    fig.suptitle('KDE Feature Visualization for Binary Classification Data', fontsize=16)
    # 绘制单个特征的KDE
    sns.kdeplot(class0_feature, ax=axes[0], label='Normal Sample', fill=True, alpha=0.5, color='#ffcc2f')
    sns.kdeplot(class1_feature, ax=axes[0], label='Abnormal Sample', fill=True, alpha=0.5, color='#ef5734')
    sns.kdeplot(recon_feature, ax=axes[0], label='recon. Normal Sample', fill=True, alpha=0.5, color='#2baf2b')
    axes[0].set_xlabel(f"Value of {feature_name}")
    axes[0].set_ylabel('Density')
    axes[0].legend()

    sns.kdeplot(class0_feature, ax=axes[1], label='Normal Sample', fill=True, alpha=0.5, color='#ffcc2f')
    sns.kdeplot(class1_feature, ax=axes[1], label='Abnormal Sample', fill=True, alpha=0.5, color='#ef5734')
    sns.kdeplot(recon_abfeature, ax=axes[1], label='recon Abnormal Sample', fill=True, alpha=0.5, color='#2baf2b')
    axes[1].set_xlabel(f"Value of {feature_name}")
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()

    np.savez_compressed(f'{save_name}_KDE_{feature_name}.npz', feat_data=data_scaled, label=labels)
    plt.savefig(f"{save_name}_{feature_name}.png")
    print(f"KDE_Data is saved in {save_name}_{feature_name} and {save_name}_KDE_{feature_name}.npz")
    plt.close()