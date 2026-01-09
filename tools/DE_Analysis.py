import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import statsmodels.stats.multitest as multi
from DetectOutlier.utils.data import get_distance, get_scale, get_global
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib
matplotlib.rcParams['font.size'] = 18  # 例如，设置全局字体大小为14

def load_data(data_name, remove_list = ()):
    dir_root = f"../datasets/Format"
    data = pd.read_csv(f"{dir_root}/{data_name}.csv", index_col=0)
    Protein_name = data.columns.values
    Sample_name = data.index.values
    Proteomic_data = get_scale(
        data, "qnorm",
        r_user="/home/server/anaconda3/envs/ppOD/lib/python3.8/site-packages/rpy2",
        r_home="/home/server/anaconda3/envs/ppOD/lib/R"
    )
    Proteomic_data = pd.DataFrame(
        data=Proteomic_data,
        index=Sample_name,
        columns=Protein_name
    )
    Proteomic_data = get_global(Proteomic_data, strategy="global_min")

    return Proteomic_data.values, Protein_name, Sample_name

def protein_ttest(control_data, experimental_data, P_protein_name, T_protein_name):
    """
    执行蛋白质组数据的t检验和多重检验校正
    https://www.cnblogs.com/leezx/p/7132099.html

    参数:
    control_data -- 对照组数据，形状为(n_proteins, n_samples)
    experimental_data -- 实验组数据，形状与control_data相同
    equal_var -- 是否假设方差齐性(True为t检验，False为Welch检验)
    correction_method -- 多重检验校正方法

    返回:
    results -- 包含检验结果的字典
    """
    n_proteins = control_data.shape[0]
    p_values = np.zeros(n_proteins)
    fold_changes = np.zeros(n_proteins)
    log_fold_changes = np.zeros(n_proteins)

    # 计算每个蛋白质的统计量
    for i in range(n_proteins):
        assert P_protein_name[i] == T_protein_name[i]
        # 计算Fold Change
        control_mean = np.mean(control_data[i])
        experimental_mean = np.mean(experimental_data[i])

        # 避免除零错误
        if control_mean == 0:
            control_mean = 1e-10

        fold_changes[i] = experimental_mean / control_mean
        log_fold_changes[i] = np.log2(fold_changes[i])

        #
        equal_val = stats.levene(experimental_data[i], control_data[i])[1]
        if equal_val < 0.05:
            # 执行t检验
            t_stat, p_val = stats.ttest_ind(
                experimental_data[i],
                control_data[i],
                equal_var=True,
                # alternative='greater'
            )
            p_values[i] = p_val
        else:
            t_stat, p_val = stats.ttest_ind(
                experimental_data[i],
                control_data[i],
                equal_var=False,
                # alternative='greater'
            )
            p_values[i] = p_val
    # 多重检验校正
    rejected, corrected_p_values, _, _ = multi.multipletests(
        p_values,
        method='fdr_bh'
    )

    BH_Pvalues = np.array(corrected_p_values)
    FC = np.array(log_fold_changes)
    BH_Pvalues_mask = (BH_Pvalues < 0.01) & (BH_Pvalues > 0)
    log2_FC_mask = (np.abs(FC) > 2)
    final_mask = BH_Pvalues_mask & log2_FC_mask
    print(f"protein length: {len(P_protein_name)} vs DE:{len(P_protein_name[final_mask])}")



    # 准备结果
    results = {
        'protein_name': P_protein_name,
        'fold_change': fold_changes,
        'log2_fold_change': log_fold_changes,
        'p_value': p_values,
        'adjusted_p_value': corrected_p_values,
        'significant': rejected,
        'DE_protein_name': P_protein_name[final_mask],
        'final_mask': final_mask,
    }

    return results


def LiearnRegression(x, y):
    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    # R²是r_value的平方
    r_squared = r_value ** 2
    print(f"slope:{slope}\t intercept:{intercept}\t r_squared:{r_squared}")


    # 使用模型进行预测
    # 绘制图形
    plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.gca()
    # 绘制真实数据散点
    plt.scatter(x, y, color='dimgray', alpha=0.7, s=5)
    y_pred = slope * x + intercept

    # 绘制真实关系线（如果知道真实关系）
    # 绘制拟合直线
    plt.plot(x, y_pred, color='crimson', linewidth=3)

    # 添加图例、标题和标签
    # plt.legend(fontsize=12)
    # plt.xlabel('After outlier removing', fontsize=14)
    # plt.ylabel('Before outlier removing', fontsize=14)

    x_major_locator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 60)
    y_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 60)

    # x_major_locator = MultipleLocator(2)
    # ax.xaxis.set_major_locator(x_major_locator)
    # y_major_locator = MultipleLocator(2)
    # ax.yaxis.set_major_locator(y_major_locator)

    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # 添加网格
    # plt.grid(True, linestyle='--', alpha=0.7)

    # 添加R²值文本
    # plt.text(0.8, 0.75, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
    #          fontsize=14, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    #          )
    ax.tick_params(direction='out', width=3, length=8)
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    return slope, intercept, r_squared

def main_run(dataset):

    P_data, P_protein_name, P_sample_name = load_data(dataset[0])
    T_data, T_protein_name, T_sample_name = load_data(dataset[1])
    assert len(P_protein_name) == len(T_protein_name)
    print("========================== Before QC ==========================")
    result = protein_ttest(P_data.T, T_data.T, P_protein_name, T_protein_name)
    print("========================== After QC ==========================")
    P_sample_mask = np.in1d(P_sample_name, outlier_info[dataset[0]])
    T_sample_mask = np.in1d(T_sample_name, outlier_info[dataset[1]])
    QC_P_data = P_data[~P_sample_mask]
    QC_T_data = T_data[~T_sample_mask]
    QC_result = protein_ttest(QC_P_data.T, QC_T_data.T, P_protein_name, T_protein_name)
    intersection_set = np.array(list(set(result['DE_protein_name']) & set(QC_result['DE_protein_name'])))
    print(f"Intersection: {len(intersection_set)}")
    common_mask= result['final_mask'] & QC_result['final_mask']
    # slope, intercept, r_squared = LiearnRegression(-np.log10(result['adjusted_p_value'][common_mask]), -np.log10(QC_result['adjusted_p_value'][common_mask]))

    slope, intercept, r_squared = LiearnRegression(np.abs(result['log2_fold_change'][common_mask]), np.abs(QC_result['log2_fold_change'][common_mask]))

if __name__ == '__main__':
    '''
 ['LADC.06N' 'LADC.43N' 'LADC.67N' 'LADC.92N']
[128.69599795 131.13830282 131.43971065 158.07419826]
    
 ['LADC.03T' 'LADC.04T' 'LADC.26T' 'LADC.90T']
[128.15861335 124.38575743 152.26457608 134.52752552]

 ['X12287T' 'X1240T' 'X424T' 'XL06T']
[110.87072272 141.82759948 130.88760527 114.34558496]

HCC_P:
    ['X12287P' 'X314P'          'X336P'      'X342P']
    [121.47730483 126.41366751 127.23109695 125.37163377]
    '''
    # MCOD
    outlier_info = {
        'HCC_P': ['X314P', 'X336P'],
        'HCC_T': ['X1240T', 'X424T'],

        'LADC_T': ['LADC.03T', 'LADC.26T', 'LADC.90T'],
        'LADC_N': [ 'LADC.43N', 'LADC.67N', 'LADC.92N'],
        # slope: 1.009699947868244     intercept: 0.02484170504613914     r_squared: 0.9913280129834566
        # slope: 1.0155549883365378     intercept: -0.008885915045451664     r_squared: 0.9803924864784989
    }

    # # ResOD
    # outlier_info = {
    #     'LADC_T': ['LADC.03T', 'LADC.26T', 'LADC.90T'],
    #     'LADC_N': ['LADC.43N', 'LADC.06N', 'LADC.92N'],
    # }
    dataset = ('HCC_P', 'HCC_T')
    # dataset = ('LADC_N', 'LADC_T')
    main_run(dataset)