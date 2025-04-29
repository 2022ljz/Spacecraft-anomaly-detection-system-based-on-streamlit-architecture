import streamlit as st
import pandas as pd
import plotly.express as px

# 页面配置
st.set_page_config(layout="wide",page_title="MyModel模型介绍" ,page_icon="📊")

with st.sidebar:
    st.markdown("### 📊MyModel介绍")
# 全局样式
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid #e6e9ef;
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
}
[data-testid="stDataFrame"] table {
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    border-radius: 8px;
    margin: 15px 0;
}
[data-testid="stDataFrame"] th {
    background-color: #f8f9fa!important;
}
</style>
""", unsafe_allow_html=True)


# 带图标的标题
st.title("📊 内置模型 MyModel介绍")

cols = st.columns([0.8,0.2])
with cols[0]:
    st.subheader("模型研究背景")
    st.markdown('''
    - 在一些特定领域，一方面由于数据标签的不可知性，一方面由于标签需要技术人员手动确定进而耗费大量的人力物力成本（如航天器数据集），在这些领域上的数据异常检测任务并不适合用传统的监督学习方式，需要用无监督学习进行异常检测。
    - 基于无监督学习的异常检测流程可以概括为三步：寻找异常判据、根据异常判据确定阈值、确定异常部分。寻找异常判据和确定阈值息息相关，决定了异常检测模型寻找异常的能力。基于自编码器（AutoEncoder）架构的无监督异常检测模型通常是接收一个输入，经过一番处理，最终得到一个输出。这一类模型由于在训练阶段学习了正常数据的表示形式，因此异常判据为待检测数据在模型上的重建误差大小，认为重建误差越大，待检测数据与正常数据的运行模式偏移越大。
    - 但是这种仅仅基于重建误差大小的异常判据显然信息表征不够丰富，因此许多即便是在时序周期预测领域表现优秀的模型，在无监督异常检测任务上成绩平平。MyModel模型旨在通过一种全新的异常检测方式，即从频域定位异常，再结合强化异常样本和正常样本间的差异，大大提高异常判据对于异常数据的区分度，从而在无监督异常检测任务上表现出强大能力。
    ''')
# 核心创新点和技术优势
st.markdown("---")
cols = st.columns(2)
with cols[0]:
    st.subheader("核心创新点")
    st.markdown("""
    <div style='color:#2a79a7; font-size:18px'>
    ▸ 采用滤波分频重建误差机制，从频域定位异常所在<br>
    ▸ 引入被动关注机制，大幅强化异常特征<br>
    ▸ 采用傅里叶优化模块取代注意力机制，更高效地全局优化数据<br>
    ▸ 通过强化阈值选择模块确定对异常更加敏感的阈值
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("### 技术优势")
    st.markdown("""
    <div style='color:#2ea8a4; font-size:16px'>
    ✓ 极高的异常召回率<br>
    ✓ 更快的模型训练速度<br> 
    ✓ 更强的综合异常检测能力<br>
    ✓ 可以处理更长更高的时间序列
    </div>
    """, unsafe_allow_html=True)

# 架构图解
st.markdown("---")
st.subheader("MyModel模型架构示意图")

# 创建左右两列布局（3:7比例）
cols = st.columns([0.4,0.1,0.5])

# 左侧列显示模型图
with cols[0]:
    st.image("pages/MyModel.png",
             use_container_width=True)

# 右侧列显示可折叠的模块说明

with cols[2]:
    st.markdown("""
    <style>
    /* 精确选择器组合 */
    div[data-testid="stExpander"] > details > summary div {
        font-size: 20px !important;
        font-weight: 900 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.expander("🟡 Filtering-Block 分频滤波模块", expanded=False):
        st.markdown("""
        **将经过FFT处理的时序数据在频域上划分为能量相等的三部分：低频、中频、高频。然后用其中两部分频域重建时序信号，得到三种缺频信号：高频缺失信号、中频缺失信号、低频缺失信号**
        - 采用了全新的异常定位，在频域定位异常
        - 等能量划分频域尽可能消除不同频域能量不同对重建误差的影响
        - 提高模型的信息表征能力
        """)

    with st.expander("🟣 Passive-Adjustment 被动关注模块", expanded=False):
        st.markdown("""
        **该模块旨在根据不同缺频信号的重建效果来动态调控模型对于三种缺频信号更新学习的关注度，保证模型投入更多注意力用于重建解耦关系稳定的频域，同时对于解耦关系不稳定的易变频域不去过度优化，为异常捕捉保留空间；同时对训练过程中的最佳信号进行计数**
        - 动态调整更新策略确保模型对正常部分不“欠拟合”，对异常部分不“过拟合”
        - 在三种缺频信号重建损失小时采用轮流大小权重更新
        - 为阈值确定部分提供了全新判据
        """)

    with st.expander("🔴 FFT-Block 傅里叶优化模块", expanded=False):
        st.markdown("""
        **借鉴FEDformer模型中处理数据的方法，在保留多头学习的基础上，采用傅里叶优化模块取代传统的注意力机制，在频域上从全局同时调整每个频率分量的相位与振幅，实现更高效的数据学习**
        - 保留传统注意力机制全局数据学习的优点
        - 对每个频率分量两个物理参数的同时优化调整
        - 更快的训练速度
        - 更强的训练效果
        """)

    with st.expander("🔵 Enhanced-Metrix 异常强化模块", expanded=False):
        st.markdown("""
        **根据被动关注模块的计数信息以及三种缺频信号的训练效果，构建异常强化矩阵，确保无论异常出现在解耦关系稳定的频域还是解耦关系不稳定的易变频域异常强化矩阵都能够有针对性地放大异常数据，提高阈值对异常的敏感性**
        - 根据模型信息构建异常强化矩阵
        - 有针对性地放大异常数据，提高阈值对于异常的敏感性
        - 解决单纯基于重建的无监督异常检测模型异常判据不足的痛点
        """)

# 性能对比部分
st.markdown("---")
st.subheader("异常检测任务表现")

# 准备数据
metrics = ["Accuracy", "Precision", "Recall", "F-score"]
model_colors = {
    "Transformer": "#1F77B4",
    "FEDformer": "#FF7F0E",
    "LSTM": "#2CA02C",
    "MyModel": "#D62728"
}

def build_dataset_df(dataset_name):
    data_map = {
        "MSL": {
            "Transformer": [0.9633, 0.8969, 0.7366, 0.8089],
            "FEDformer": [0.9657, 0.9065, 0.7522, 0.8222],
            "LSTM": [0.9327, 0.9136, 0.6924, 0.7878],
            "MyModel": [0.9895, 0.9411, 0.9601, 0.9505]
        },
        "SMAP": {
            "Transformer": [0.9427, 0.9090, 0.6138, 0.7328],
            "FEDformer": [0.9353, 0.9018, 0.5584, 0.6870],
            "LSTM": [0.9487, 0.8512, 0.8465, 0.8488],
            "MyModel": [0.9899, 0.9322, 0.9932, 0.9617]
        }
    }
    records = []
    for model, values in data_map[dataset_name].items():
        for idx, metric in enumerate(metrics):
            records.append({
                "Dataset": dataset_name,
                "Model": model,
                "Metric": metric,
                "Value": values[idx]
            })
    return pd.DataFrame(records)

def create_comparison_chart(df, title):
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        color_discrete_map=model_colors,
        text_auto=".2%",
        height=400,
        title=title
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_range=[0.5, 1.05],
        xaxis_title="评估指标",
        yaxis_title="指标数值",
        legend_title="模型类型",
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(tickangle=-45)
    )
    return fig

# 分列布局
col1, col2 = st.columns(2)

# MSL列
with col1:
    msl_df = build_dataset_df("MSL")
    st.plotly_chart(
        create_comparison_chart(msl_df, "MSL数据集性能对比"),
        use_container_width=True
    )

    st.markdown("##### MSL数据集指标详情")
    msl_pivot = msl_df.pivot_table(
        index="Model",
        columns="Metric",
        values="Value"
    )
    # 自定义行列顺序
    column_order = ['Accuracy','Precision','Recall','F-score']  # 新列顺序
    row_order = ['MyModel', 'FEDformer', 'LSTM', 'Transformer']  # 新行顺序

    msl_pivot = msl_pivot[column_order]  # 调整列顺序
    msl_pivot = msl_pivot.reindex(row_order)  # 调整行顺序

    st.dataframe(
        msl_pivot.style.format("{:.2%}")
        .highlight_max(axis=0, color="#d4edda")
        .applymap(lambda x: '', subset=pd.IndexSlice['MyModel', :])
        .set_properties(subset=pd.IndexSlice['MyModel', :],
                        **{'background-color': '#fff3cd'}),
        use_container_width=True
    )

# SMAP列
with col2:
    smap_df = build_dataset_df("SMAP")
    st.plotly_chart(
        create_comparison_chart(smap_df, "SMAP数据集性能对比"),
        use_container_width=True
    )

    st.markdown("##### SMAP数据集指标详情")
    smap_pivot = smap_df.pivot_table(
        index="Model",
        columns="Metric",
        values="Value"
    )
    # 自定义行列顺序
    column_order = ['Accuracy','Precision','Recall','F-score']  # 新列顺序
    row_order = ['MyModel', 'FEDformer', 'LSTM', 'Transformer']  # 新行顺序

    smap_pivot = smap_pivot[column_order]  # 调整列顺序
    smap_pivot = smap_pivot.reindex(row_order)  # 调整行顺序
    st.dataframe(
        smap_pivot.style.format("{:.2%}")
        .highlight_max(axis=0, color="#d4edda")
        .applymap(lambda x: '', subset=pd.IndexSlice['MyModel', :])
        .set_properties(subset=pd.IndexSlice['MyModel', :],
                        **{'background-color': '#fff3cd'}),
        use_container_width=True
    )