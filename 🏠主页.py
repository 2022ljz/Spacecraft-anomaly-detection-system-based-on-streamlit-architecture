import streamlit as st

st.set_page_config(
    page_title="欢迎使用航天器异常检测系统！",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠欢迎使用航天器异常检测系统！")
st.sidebar.success("这里是主页")
col1,_ = st.columns([0.8,0.2])
with col1:
    # 主页面内容
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

    content = """
    <div class='big-font'>

    随着航天技术的迅猛发展，航天器在轨道飞行中的遥测数据（Telemetry Value）成为监测其健康状态和性能至关重要的信息来源。遥测数据记录了航天器在飞行过程中的各类状态参数，如温度、压力、电池电量、姿态等。这些数据不仅用于日常监控，还能为航天器的故障诊断和维护决策提供有力支持。

    **面临的挑战：**  
    由于传感器故障、数据丢失、外部环境影响等原因，遥测数据中可能会出现异常。这些异常数据若不及时识别和处理，可能会对航天器的安全和任务成功造成严重影响。当前监测系统仅能覆盖部分异常类型，且系统开发和维护需要高度专业的技术人员。

    **我们的解决方案：**  
    本系统采用来自NASA的SMAP和MSL数据集，通过自主研发的基于Encoder架构的无监督异常检测模型**MyModel**，为航天器遥测数据提供智能化的异常检测服务。系统具备以下特点：
    - 支持多维度时序数据分析，兼容NASA标准数据集
    - 无监督学习动态阈值确定适应新型异常模式
    - 提供可视化分析界面，直观感受异常检测结果
    - 采用新的异常定位方法，拥有极高异常召回率

    通过自动化检测系统，有效减轻工程师和专家的数据分析负担，助力航天任务的安全执行。
    </div>
    """

    st.markdown(content, unsafe_allow_html=True)

    st.image("pages/fly.png",
             use_container_width=True)


def nav_card(title, description, icon):
    return f"""
    <div style="
        padding: 1.5rem;
        margin: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        min-height: 150px;
    ">
        <div style="font-size: 2rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: #0d6efd; margin: 0 0 0.5rem 0;">{title}</h3>
        <p style="color: #6c757d; line-height: 1.5;">{description}</p>
    </div>
    """

st.markdown("""
<style>
.nav-header {
    font-size: 1.2rem !important;
    color: #495057 !important;
    margin: 2rem 0 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="nav-header">📌 在侧边栏切换不同页面，探索更多内容</p>', unsafe_allow_html=True)

cols = st.columns(3)
with cols[0]:
    st.markdown(nav_card(
        "航天数据集可视化",
        "SMAP/MSL数据集原始数据信号的可视化展示，并提供数据摘要与下载功能",
        "📈"
    ), unsafe_allow_html=True)

with cols[1]:
    st.markdown(nav_card(
        "内置模型 MyModel介绍",
        "全新的基于Encoder架构的无监督异常检测模型，展示模型架构示意图与重要模块介绍",
        "📊"
    ), unsafe_allow_html=True)

with cols[2]:
    st.markdown(nav_card(
        "异常检测分析",
        "对数据集进行异常检测以及结果可视化，提供检测指标与下载功能",
        "🔍"
    ), unsafe_allow_html=True)


