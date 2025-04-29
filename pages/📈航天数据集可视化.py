import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import time

# 页面配置
st.set_page_config(
    page_title="航天数据集可视化",
    page_icon="📈",
    layout="wide"
)

# 标题区域
st.title('📈航天数据集可视化')
cols = st.columns([0.8,0.2])
with cols[0]:
    st.subheader('功能介绍')
    st.markdown("""
    - 用户可以看到来自NASA的MSL和SMAP数据集的可视化效果。为了拟合在无监督异常检测任务中的处理，让用户可以更直观感受模型的"输入"，本页面已经提前将两个数据集中的各个通道合并。MSL数据集有25个维度，SMAP数据集有55个维度，遥测数据位于第一维度上，其余维度均是相关命令的发送与否，1表示已发送，0表示未发送。本页面的可视化功能是对两个数据集中的第一维度进行可视化。
    - 由于数据集中的数据点过多，本页面采用动态窗口可视化展示，用户可以在侧边栏的"配置面板"处进行相关配置。
    - 本页面支持数据导出功能，可以对数据集进行下载，但是不包含数据标签。下载的数据文件为csv文件。
    """)
st.markdown("---")

# 数据集路径配置
DATASET_PATHS = {
    "SMAP": "dataset/SMAP/SMAP_train.npy",
    "MSL": "dataset/MSL/MSL_train.npy"
}

# 初始化session状态（结构化初始化）
STATE_SCHEMA = {
    'plot_created': False,
    'current_pos': 0,
    'progress': 0,
    'loading': False,
    'applied_params': {
        "dataset": "SMAP",
        "window_size": 1000,
        "start_idx": 0,
        "end_idx": 10000
    }
}

for key, default in STATE_SCHEMA.items():
    if isinstance(default, dict):
        st.session_state.setdefault(key, default.copy())
    else:
        st.session_state.setdefault(key, default)

# 侧边栏控件（卡片式布局）
with st.sidebar:
    st.markdown("### 🛠️ 配置面板")

    with st.expander("📂 数据源设置", expanded=True):
        selected_dataset = st.selectbox(
            "选择数据集",
            options=["SMAP", "MSL"],
            help="SMAP: 土壤湿度卫星数据 | MSL: 火星科学实验室数据"
        )

    with st.expander("⚙️ 可视化参数", expanded=True):
        try:
            current_data = np.load(DATASET_PATHS[selected_dataset], allow_pickle=True)
            current_real_data = current_data[:, 0]
            current_max_length = len(current_real_data)
        except Exception as e:
            st.error(f"🚨 数据集加载失败: {str(e)}")
            st.stop()

        selected_window_size = st.slider(
            "窗口长度",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="可视化窗口包含的数据点数"
        )

        MAX_RANGE = current_max_length - 1
        selected_range = st.slider(
            "数据范围",
            min_value=0,
            max_value=MAX_RANGE,
            value=(
                min(st.session_state.applied_params["start_idx"], MAX_RANGE),
                min(st.session_state.applied_params["end_idx"], MAX_RANGE)
            ),
            step=1,
            format="%d"
        )

    apply_button = st.button(
        "🚀 应用配置",
        use_container_width=True,
        type="primary"
    )

# 主界面布局
progress_placeholder = st.empty()
if apply_button:
    if selected_range[0] >= selected_range[1]:
        st.error("❌ 起始位置必须小于结束位置！")
        st.stop()

    st.session_state.applied_params = {
        "dataset": selected_dataset,
        "window_size": selected_window_size,
        "start_idx": selected_range[0],
        "end_idx": selected_range[1]
    }
    st.session_state.current_pos = selected_range[0]
    st.session_state.loading = True
    st.session_state.plot_created = False
    st.rerun()

# 处理加载状态
if st.session_state.loading:
    with progress_placeholder.container():
        # 显示加载转圈
        with st.spinner("🚀 正在生成可视化图表，请稍候..."):
            time.sleep(1)

            # 数据处理完成后设置以下状态
            st.session_state.plot_created = True

    # 关闭加载状态并清空占位符
    st.session_state.loading = False
    progress_placeholder.empty()
    st.rerun()

# 主可视化模块
if st.session_state.plot_created:
    try:
        # 数据统计卡片
        with st.container():
            col_info, col_viz = st.columns([1, 3])

            with col_info:
                st.subheader("📋 数据集摘要")
                applied_data = np.load(DATASET_PATHS[st.session_state.applied_params["dataset"]], allow_pickle=True)
                applied_real_data = applied_data[:, 0]
                max_length = len(applied_real_data)

                st.markdown(f"""
                <div style="
                    padding: 1rem;
                    background: #FFFFFF;
                    border-radius: 8px;
                    border: 2px solid #1E3A8A;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    color: #1E3A8A;
                ">
                    <p style="margin: 0.5rem 0; color: #1E3A8A;">🏷️ <strong>数据集名称:</strong> {st.session_state.applied_params["dataset"]}</p>
                    <p style="margin: 0.5rem 0; color: #1E3A8A;">📏 <strong>总数据量:</strong> {max_length:,} 点</p>
                    <p style="margin: 0.5rem 0; color: #1E3A8A;">🎯 <strong>分析范围:</strong> {st.session_state.applied_params["start_idx"]}-{st.session_state.applied_params["end_idx"]}</p>
                    <p style="margin: 0.5rem 0; color: #1E3A8A;">🖼️ <strong>窗口大小:</strong> {st.session_state.applied_params["window_size"]}</p>
                </div>
                """, unsafe_allow_html=True)

                # 统计指标
                subset_data = applied_real_data[
                              st.session_state.applied_params["start_idx"]:st.session_state.applied_params["end_idx"]
                              ]
                stats = {
                    "mean": np.mean(subset_data),
                    "std": np.std(subset_data),
                    "min": np.min(subset_data),
                    "max": np.max(subset_data)
                }
                st.write('')

                st.markdown("""
                 <div style="
                    padding: 1rem;
                    background: #FFFFFF;
                    border-radius: 8px;
                    border: 2px solid #1E3A8A;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    color: #1E3A8A;
                ">
                    <h4 style="margin-top: 0;">📐 统计特征</h4>
                    <p style="margin: 0.5rem 0;">📌 均值: {mean:.4f}</p>
                    <p style="margin: 0.5rem 0;">📏 标准差: {std:.4f}</p>
                    <p style="margin: 0.5rem 0;">⬇️ 最小值: {min:.4f}</p>
                    <p style="margin: 0.5rem 0;">⬆️ 最大值: {max:.4f}</p>
                </div>
                """.format(**stats), unsafe_allow_html=True)

            with col_viz:
                st.subheader(f'📈{st.session_state.applied_params["dataset"]} 时序可视化')
                # 交互式图表
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=np.arange(st.session_state.current_pos,
                                st.session_state.current_pos + st.session_state.applied_params["window_size"]),
                    y=applied_real_data[
                      st.session_state.current_pos:st.session_state.current_pos + st.session_state.applied_params[
                          "window_size"]
                      ],
                    mode='lines',
                    line=dict(color='#4CAF50', width=1.5),
                    name='传感器数据'
                ))

                fig.update_layout(
                    xaxis_title='数据点索引',
                    yaxis_title='传感器读数',
                    height=600,
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)

                # 导航控制条
                with st.container():
                    min_pos = st.session_state.applied_params["start_idx"]
                    max_pos = st.session_state.applied_params["end_idx"] - st.session_state.applied_params[
                        "window_size"]
                    if max_pos < min_pos:
                        max_pos = min_pos

                    new_pos = st.slider(
                        "导航位置",
                        min_value=min_pos,
                        max_value=max_pos,
                        value=st.session_state.current_pos,
                        step=st.session_state.applied_params["window_size"] // 10,
                        format="%d",
                        key='nav_slider'
                    )

                    if new_pos != st.session_state.current_pos:
                        st.session_state.current_pos = new_pos
                        st.rerun()

                # 数据下载面板
                with st.expander("⬇️ 数据导出", expanded=True):
                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        # 完整数据集下载
                        try:
                            full_data = np.load(
                                DATASET_PATHS[st.session_state.applied_params["dataset"]],
                                allow_pickle=True
                            )
                            df_full = pd.DataFrame(full_data)
                            df_full.columns = ["value"] + [f"col_{i}" for i in range(1, df_full.shape[1])]

                            st.download_button(
                                label="下载完整数据集",
                                data=df_full.to_csv(index=False).encode('utf-8'),
                                file_name=f"{st.session_state.applied_params['dataset']}_full.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="包含所有传感器通道的完整数据集"
                            )
                        except Exception as e:
                            st.error(f"导出失败: {str(e)}")

                    with col_dl2:
                        # 范围数据下载
                        try:
                            subset_data = applied_real_data[
                                          st.session_state.applied_params["start_idx"]:st.session_state.applied_params[
                                              "end_idx"]
                                          ]
                            df_subset = pd.DataFrame(subset_data)
                            df_subset.columns = ["value"] + [f"col_{i}" for i in range(1, df_subset.shape[1])]

                            st.download_button(
                                label="下载当前范围数据",
                                data=df_subset.to_csv(index=False).encode('utf-8'),
                                file_name=f"{st.session_state.applied_params['dataset']}_{st.session_state.applied_params['start_idx']}-{st.session_state.applied_params['end_idx']}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="仅包含选定范围的数据切片"
                            )
                        except Exception as e:
                            st.error(f"导出失败: {str(e)}")

    except Exception as e:
        st.error(f"💥 运行时错误: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ 请先在侧边栏完成参数配置并点击【应用配置】")
