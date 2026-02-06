"""
PTA期货交易策略回测系统 - 实战逻辑展示终端
==========================================

基于PX原料利润"领先效应"的PTA期货交易策略回测系统
将回测结果转化为直观的投资决策报告

运行方式：
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
import sys
from datetime import datetime, timedelta
from matplotlib import font_manager
import matplotlib
import urllib.request
import os
import tempfile

# 配置matplotlib中文字体（兼容Streamlit Cloud）
def download_chinese_font():
    """下载中文字体文件到临时目录（用于Streamlit Cloud）"""
    # 使用 Noto Sans CJK SC 字体（Google开源中文字体）
    # 使用 GitHub 上的 TTF 字体文件（更可靠）
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosanscjksc/NotoSansCJKsc-Regular.otf"
    
    # 如果URL是OTF格式，尝试转换为TTF或直接使用
    # 实际上，matplotlib支持OTF格式，所以可以直接使用
    font_name = "NotoSansCJKsc-Regular.otf"
    
    # 获取matplotlib字体目录
    try:
        # 尝试使用matplotlib的字体缓存目录
        cache_dir = font_manager.get_cachedir()
        font_dir = Path(cache_dir).parent / "fonts" / "ttf"
    except:
        # 如果失败，使用临时目录
        font_dir = Path(tempfile.gettempdir()) / "matplotlib_fonts"
    
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / font_name
    
    # 如果字体文件已存在，直接返回
    if font_path.exists() and font_path.stat().st_size > 0:
        return str(font_path)
    
    # 尝试下载字体文件
    try:
        # 设置超时时间（10秒）
        with urllib.request.urlopen(font_url, timeout=10) as response:
            with open(font_path, 'wb') as f:
                f.write(response.read())
        
        # 验证文件是否下载成功
        if font_path.exists() and font_path.stat().st_size > 1000:  # 至少1KB
            # 清除matplotlib字体缓存，强制重新加载
            try:
                # 将字体文件添加到matplotlib的字体路径
                font_manager.fontManager.addfont(str(font_path))
                font_manager._rebuild()
            except:
                pass
            return str(font_path)
    except Exception as e:
        # 下载失败，返回None
        return None
    
    return None

def setup_chinese_font():
    """配置matplotlib中文字体，兼容Streamlit Cloud环境"""
    # 尝试使用系统中文字体（按优先级排序）
    chinese_fonts = [
        "Microsoft YaHei", "Microsoft YaHei UI", 
        "SimHei", "SimSun", "KaiTi", "FangSong",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans CJK TC",
        "Source Han Sans CN", "Source Han Sans SC",
        "STHeiti", "STSong", "STKaiti", "STFangsong"
    ]
    
    # 获取所有可用字体
    try:
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    except:
        available_fonts = []
    
    # 查找可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 清除matplotlib字体缓存，强制重新加载
                try:
                    font_manager._rebuild()
                except:
                    pass
                return font_manager.FontProperties(family=font)
            except Exception as e:
                continue
    
    # 如果找不到中文字体，尝试下载字体文件（用于Streamlit Cloud）
    downloaded_font_path = download_chinese_font()
    if downloaded_font_path:
        try:
            # 使用下载的字体文件
            font_prop = font_manager.FontProperties(fname=downloaded_font_path)
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            # 清除matplotlib字体缓存，强制重新加载
            try:
                font_manager._rebuild()
            except:
                pass
            return font_prop
        except Exception as e:
            pass
    
    # 如果都失败，尝试使用系统默认sans-serif字体
    plt.rcParams['axes.unicode_minus'] = False
    try:
        default_font = font_manager.FontProperties()
        plt.rcParams['font.sans-serif'] = ['sans-serif']
        return default_font
    except:
        # 最后的fallback：使用DejaVu Sans（虽然不支持中文，但至少不会报错）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return font_manager.FontProperties(family='DejaVu Sans')

# 初始化字体配置
_chinese_font_prop = setup_chinese_font()

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from strategy import (
    StrategyConfig, CONFIG,
    load_merged_data_with_basis,
    generate_signals,
    backtest_strategy,
    get_chinese_font_prop as _get_chinese_font_prop_original
)

# 重写get_chinese_font_prop函数，优先使用我们配置的字体
def get_chinese_font_prop():
    """获取中文字体属性对象（优先使用全局配置的字体）"""
    if _chinese_font_prop is not None:
        return _chinese_font_prop
    # 如果全局配置失败，尝试使用strategy.py中的函数
    result = _get_chinese_font_prop_original()
    if result is not None:
        return result
    # 如果都失败，尝试使用系统默认字体
    try:
        # 使用matplotlib的默认字体配置
        default_prop = font_manager.FontProperties()
        # 确保rcParams已正确设置
        if 'font.sans-serif' not in plt.rcParams or not plt.rcParams['font.sans-serif']:
            plt.rcParams['font.sans-serif'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return default_prop
    except:
        # 最后的fallback：使用DejaVu Sans（虽然不支持中文，但至少不会报错）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return font_manager.FontProperties(family='DejaVu Sans')

warnings.filterwarnings("ignore")

# 页面配置
st.set_page_config(
    page_title="PTA期货策略实战逻辑展示终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .logic-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .logic-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .logic-card-yellow {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .logic-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .profit-highlight {
        background-color: #d4edda !important;
    }
    .loss-highlight {
        background-color: #f8d7da !important;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        height: 3.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<div class="main-header">📈 PTA期货策略实战逻辑展示终端</div>', unsafe_allow_html=True)

# ============================================================================
# 核心逻辑卡片（在回测按钮上方）
# ============================================================================
st.markdown("### 🎯 策略核心逻辑")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="logic-card logic-card-green">
        <h3>🟢 PX 动力</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
        监测上游原料利润，预判 PTA 涨跌动力<br>
        <small>当PX原料利润变动超过日常波动的1.5倍时，说明成本端在推涨</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="logic-card logic-card-yellow">
        <h3>🟡 加工费安全垫</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
        寻找 PTA 厂长亏损严重的时刻，确保入场安全<br>
        <small>只在加工费低于450元/吨时做多，避免高位接盘</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="logic-card logic-card-blue">
        <h3>🔵 18天传导周期</h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
        基于历史统计，给利润释放留出充足时间<br>
        <small>持仓15天左右，等待成本传导带来的价格上涨</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# 侧边栏 - 参数配置（使用expander分组）
# ============================================================================
st.sidebar.header("⚙️ 策略参数配置")

# 数据文件上传
st.sidebar.subheader("📁 数据文件")
uploaded_file = st.sidebar.file_uploader(
    "上传数据文件（CSV格式）",
    type=['csv'],
    help="⚠️ 重要：必须包含'期货价格'列（futures_price或主力合约期货价格），不是现货价格！"
)

if uploaded_file is not None:
    data_path = uploaded_file
else:
    data_path = None
    st.sidebar.warning("⚠️ 请上传数据文件")

# 基础参数（默认显示）
st.sidebar.subheader("💰 基础设置")
initial_capital = st.sidebar.number_input(
    "初始资金（元）",
    min_value=100000,
    max_value=10000000,
    value=CONFIG.INITIAL_CAPITAL,
    step=100000,
    format="%d",
    key="initial_capital"
)
st.sidebar.caption("💡 回测的起始资金")

# 回测时间段选择（如果有数据）
backtest_start_date = None
backtest_end_date = None

if 'df' in st.session_state and st.session_state['df'] is not None:
    df_temp = st.session_state['df']
    if len(df_temp) > 0:
        min_date = df_temp['date'].min()
        max_date = df_temp['date'].max()
        st.sidebar.info(f"📅 数据时间范围：\n{min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')}")
        
        # 添加时间段选择器
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⏰ 回测时间段选择")
        use_custom_range = st.sidebar.checkbox(
            "自定义回测时间段",
            value=False,
            help="勾选后可以自定义回测的开始和结束时间",
            key="use_custom_range"
        )
        
        if use_custom_range:
            # 将日期转换为datetime类型（如果还不是）
            if isinstance(min_date, pd.Timestamp):
                min_date_val = min_date.to_pydatetime().date()
            else:
                min_date_val = pd.to_datetime(min_date).date()
            
            if isinstance(max_date, pd.Timestamp):
                max_date_val = max_date.to_pydatetime().date()
            else:
                max_date_val = pd.to_datetime(max_date).date()
            
            backtest_start_date = st.sidebar.date_input(
                "回测开始日期",
                value=min_date_val,
                min_value=min_date_val,
                max_value=max_date_val,
                key="backtest_start_date"
            )
            
            backtest_end_date = st.sidebar.date_input(
                "回测结束日期",
                value=max_date_val,
                min_value=min_date_val,
                max_value=max_date_val,
                key="backtest_end_date"
            )
            
            # 验证日期范围
            if backtest_start_date >= backtest_end_date:
                st.sidebar.warning("⚠️ 开始日期必须早于结束日期")
            else:
                st.sidebar.success(f"✅ 将回测 {backtest_start_date} 至 {backtest_end_date} 的数据")
        else:
            st.sidebar.caption("💡 未勾选时，将使用全部数据")

# 信号灵敏度配置（expander，默认折叠）
with st.sidebar.expander("🛠️ 信号灵敏度配置", expanded=False):
    px_atr_period = st.slider(
        "观察PX原料利润的周期（天数）",
        min_value=5,
        max_value=50,
        value=CONFIG.PX_ATR_PERIOD,
        step=5,
        help="用来计算PX原料利润日常波动剧烈程度的观察天数",
        key="px_atr_period"
    )
    st.caption("💡 建议20天")
    
    px_atr_multiplier = st.slider(
        "PX原料利润变动倍数",
        min_value=0.5,
        max_value=3.0,
        value=CONFIG.PX_ATR_MULTIPLIER,
        step=0.1,
        help="当PX原料利润变动超过日常波动的多少倍时，才认为是'大行情'",
        key="px_atr_multiplier"
    )
    st.caption("💡 1.5倍是平衡点")

# 安全垫过滤器（expander，默认折叠）
with st.sidebar.expander("🛡️ 安全垫过滤器", expanded=False):
    enable_margin_filter = st.checkbox(
        "启用安全垫过滤器",
        value=CONFIG.ENABLE_MARGIN_FILTER,
        help="只在PTA生产利润足够低时才做多",
        key="enable_margin_filter"
    )
    
    if enable_margin_filter:
        margin_long = st.number_input(
            "做多安全垫阈值（元/吨）",
            min_value=0,
            max_value=1000,
            value=CONFIG.MARGIN_LONG_THRESHOLD,
            step=10,
            key="margin_long"
        )
        st.caption("💡 建议450元/吨")
        
        margin_short = st.number_input(
            "做空安全垫阈值（元/吨）",
            min_value=0,
            max_value=1000,
            value=CONFIG.MARGIN_SHORT_THRESHOLD,
            step=10,
            key="margin_short"
        )
        st.caption("💡 建议750元/吨")
    else:
        margin_long = CONFIG.MARGIN_LONG_THRESHOLD
        margin_short = CONFIG.MARGIN_SHORT_THRESHOLD

# 交易执行参数（expander，默认折叠）
with st.sidebar.expander("💼 交易执行参数", expanded=False):
    st.markdown("#### 💰 资金管理")
    
    # 显示当前资金（用于计算示例）
    if 'initial_capital' in st.session_state:
        current_capital = st.session_state.get('initial_capital', initial_capital)
    else:
        current_capital = initial_capital
    
    max_position_ratio = st.slider(
        "最大仓位比例（%）",
        min_value=10,
        max_value=100,
        value=int(CONFIG.MAX_POSITION_RATIO * 100),
        step=5,
        help="最多使用多少比例的总资金作为保证金（例如：100万资金×80%=80万可用保证金）",
        key="max_position_ratio"
    ) / 100
    available_margin_example = current_capital * max_position_ratio
    st.caption(f"💡 建议80%，留20%作为风险缓冲 | 当前资金{current_capital:,.0f}元 × {int(max_position_ratio*100)}% = {available_margin_example:,.0f}元可用保证金")
    
    position_size = st.slider(
        "每次投入比例（%）",
        min_value=1,
        max_value=int(max_position_ratio * 100),
        value=int(CONFIG.POSITION_SIZE * 100),
        step=1,
        help=f"在可用保证金内，每次开仓投入多少比例（例如：{available_margin_example:,.0f}元可用保证金 × 10% = {available_margin_example*0.1:,.0f}元用于本次开仓）",
        key="position_size"
    ) / 100
    invested_margin_example = available_margin_example * position_size
    st.caption(f"💡 建议10-20% | {available_margin_example:,.0f}元可用保证金 × {int(position_size*100)}% = {invested_margin_example:,.0f}元用于本次开仓")
    
    # 详细说明
    with st.expander("📖 资金管理参数说明", expanded=False):
        st.markdown("""
        **两个参数的关系：**
        
        1. **最大仓位比例** = 总资金的安全上限
           - 例如：100万资金 × 80% = 80万可用保证金
           - 这是**所有持仓**加起来最多能用的资金
           - 建议80%，留20%作为风险缓冲
        
        2. **每次投入比例** = 单次开仓的资金比例
           - 例如：80万可用保证金 × 10% = 8万用于本次开仓
           - 这是**每次新开仓**时投入的资金
           - 建议10-20%，分散风险
        
        **计算手数的流程：**
        ```
        总资金 = 100万
        可用保证金 = 100万 × 最大仓位比例(80%) = 80万
        本次投入 = 80万 × 每次投入比例(10%) = 8万
        合约价值 = 8万 × 杠杆倍数(10倍) = 80万
        手数 = 80万 ÷ (价格6000元/吨 × 5吨/手) = 26手
        ```
        
        **为什么这样设计？**
        - 最大仓位比例：防止满仓，留出风险缓冲
        - 每次投入比例：避免单次投入过大，分散风险
        - 随着资金增长，可开手数自动增加
        """)
    
    enable_dynamic_position = st.checkbox(
        "启用分级仓位（根据加工费自动调整）",
        value=CONFIG.ENABLE_DYNAMIC_POSITION,
        key="enable_dynamic_position"
    )
    if enable_dynamic_position:
        st.caption("💡 加工费<350：仓位×1.5；加工费>600：仓位×0.5")
    
    st.markdown("---")
    st.markdown("#### ⏱️ 持仓周期")
    
    holding_period = st.slider(
        "持仓天数",
        min_value=5,
        max_value=30,
        value=CONFIG.HOLDING_PERIOD,
        step=1,
        key="holding_period"
    )
    
    st.markdown("---")
    st.markdown("#### 📊 期货交易参数")
    
    # 计算最大杠杆倍数（基于最低保证金比例）
    max_leverage = 1.0 / CONFIG.MIN_MARGIN_RATE  # 约14.3倍
    
    leverage = st.slider(
        "杠杆倍数",
        min_value=1.0,
        max_value=float(max_leverage),
        value=CONFIG.LEVERAGE,
        step=0.5,
        help=f"期货交易的杠杆倍数（1.0表示无杠杆，最高{max_leverage:.1f}倍，对应最低保证金比例{CONFIG.MIN_MARGIN_RATE*100:.0f}%）",
        key="leverage"
    )
    st.caption(f"💡 PTA期货最低保证金{CONFIG.MIN_MARGIN_RATE*100:.0f}%，最高杠杆{max_leverage:.1f}倍")
    
    # 显示实际保证金比例
    actual_margin_rate = 1.0 / leverage if leverage > 0 else 1.0
    if actual_margin_rate < CONFIG.MIN_MARGIN_RATE:
        st.warning(f"⚠️ 当前杠杆{leverage:.1f}倍对应保证金比例{actual_margin_rate*100:.2f}%，低于最低要求{CONFIG.MIN_MARGIN_RATE*100:.0f}%")
    else:
        st.info(f"✅ 当前杠杆{leverage:.1f}倍对应保证金比例{actual_margin_rate*100:.2f}%")
    
    # 显示实际能开的手数估算（基于当前资金和杠杆）
    if 'initial_capital' in st.session_state:
        current_capital = st.session_state.get('initial_capital', initial_capital)
    else:
        current_capital = initial_capital
    
    # 估算能开的手数（假设价格为6000元/吨）
    estimated_price = 6000
    available_margin = current_capital * max_position_ratio
    actual_invested = available_margin * position_size
    estimated_contract_value = actual_invested * leverage
    estimated_contracts = int(estimated_contract_value / (estimated_price * 5))
    
    if estimated_contracts > 0:
        st.info(f"📊 估算：当前资金{current_capital:,.0f}元，最多可开约{estimated_contracts}手（假设价格{estimated_price}元/吨，杠杆{leverage}倍）")
    
    # 手续费计算方式选择
    use_fixed_commission = st.checkbox(
        "使用固定手续费（推荐）",
        value=CONFIG.USE_FIXED_COMMISSION,
        help="PTA期货通常使用固定手续费，每手固定金额",
        key="use_fixed_commission"
    )
    
    if use_fixed_commission:
        # 固定手续费（每手固定金额）
        commission_per_contract = st.number_input(
            "固定手续费（元/手）",
            min_value=0.0,
            max_value=10.0,
            value=CONFIG.COMMISSION_PER_CONTRACT,
            step=0.1,
            format="%.1f",
            help="每手合约的固定手续费（开仓+平仓各收一次，共2次）",
            key="commission_per_contract"
        )
        st.caption(f"💡 PTA期货通常为3.3元/手（开仓+平仓共{commission_per_contract*2:.1f}元）")
        commission_rate = 0.0  # 不使用比例手续费
    else:
        # 比例手续费（按合约价值）
        commission_rate = st.number_input(
            "手续费率（按合约价值）",
            min_value=0.0,
            max_value=0.01,
            value=CONFIG.COMMISSION_RATE,
            step=0.0001,
            format="%.4f",
            help="手续费占合约价值的比例（如0.0001表示万分之一）",
            key="commission_rate"
        )
        st.caption("💡 通常为0.0001-0.0003（万分之一到万分之三）")
        commission_per_contract = 0.0  # 不使用固定手续费
    
    # PTA期货合约单位固定为5吨/手
    contract_size = 5
    st.markdown(f"**合约单位：** {contract_size} 吨/手（PTA期货固定）")
    st.caption("💡 PTA期货1手=5吨，不可调整")
    st.caption("💡 建议15-18天")

# 风险控制参数（expander，默认折叠）
with st.sidebar.expander("🛡️ 风险控制标准", expanded=False):
    atr_multiplier = st.slider(
        "价格波动剧烈程度倍数（止损用）",
        min_value=0.5,
        max_value=3.0,
        value=CONFIG.ATR_MULTIPLIER,
        step=0.1,
        key="atr_multiplier"
    )
    st.caption("💡 建议1.5倍")
    
    atr_period = st.slider(
        "计算价格波动剧烈程度的周期（天数）",
        min_value=5,
        max_value=30,
        value=CONFIG.ATR_PERIOD,
        step=1,
        key="atr_period"
    )
    st.caption("💡 建议14天")
    
    enable_px_ma_stop = st.checkbox(
        "启用PX价差均线止损",
        value=CONFIG.ENABLE_PX_MA_STOP,
        key="enable_px_ma_stop"
    )
    
    if enable_px_ma_stop:
        px_ma_period = st.slider(
            "PX价差均线周期（天）",
            min_value=3,
            max_value=10,
            value=CONFIG.PX_MA_PERIOD,
            step=1,
            key="px_ma_period"
        )
        st.caption("💡 建议5天")

# 止盈参数（expander，默认折叠）
with st.sidebar.expander("🎯 止盈参数", expanded=False):
    enable_basis_tp = st.checkbox(
        "启用基差止盈（现货涨不动时提前落袋）",
        value=CONFIG.ENABLE_BASIS_TAKE_PROFIT,
        key="enable_basis_tp"
    )
    
    if enable_basis_tp:
        basis_tp_threshold = st.slider(
            "止盈盈利阈值（%）",
            min_value=0.5,
            max_value=5.0,
            value=CONFIG.BASIS_TAKE_PROFIT_THRESHOLD,
            step=0.1,
            key="basis_tp_threshold"
        )
        
        basis_min_holding = st.slider(
            "基差止盈最小持仓天数",
            min_value=5,
            max_value=15,
            value=CONFIG.BASIS_MIN_HOLDING_DAYS,
            step=1,
            key="basis_min_holding"
        )
        st.caption("💡 建议7天")
        
        basis_decline_days = st.slider(
            "基差连续走弱天数",
            min_value=2,
            max_value=7,
            value=CONFIG.BASIS_DECLINE_DAYS,
            step=1,
            key="basis_decline_days"
        )
    else:
        basis_tp_threshold = CONFIG.BASIS_TAKE_PROFIT_THRESHOLD
        basis_decline_days = CONFIG.BASIS_DECLINE_DAYS
        basis_min_holding = CONFIG.BASIS_MIN_HOLDING_DAYS

# ============================================================================
# 主界面 - 回测按钮
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    run_backtest = st.button("🚀 开始回测", type="primary", use_container_width=True)

# 执行回测
if run_backtest:
    if data_path is None:
        st.error("❌ 请先上传数据文件")
        st.stop()
    
    with st.spinner("正在加载数据并执行回测..."):
        try:
            # 更新配置
            CONFIG.PX_ATR_PERIOD = px_atr_period
            CONFIG.PX_ATR_MULTIPLIER = px_atr_multiplier
            CONFIG.ENABLE_MARGIN_FILTER = enable_margin_filter
            CONFIG.MARGIN_LONG_THRESHOLD = margin_long
            CONFIG.MARGIN_SHORT_THRESHOLD = margin_short
            CONFIG.INITIAL_CAPITAL = initial_capital
            CONFIG.POSITION_SIZE = position_size
            CONFIG.MAX_POSITION_RATIO = max_position_ratio
            CONFIG.HOLDING_PERIOD = holding_period
            CONFIG.ATR_MULTIPLIER = atr_multiplier
            CONFIG.ATR_PERIOD = atr_period
            CONFIG.ENABLE_PX_MA_STOP = enable_px_ma_stop
            CONFIG.PX_MA_PERIOD = px_ma_period
            CONFIG.ENABLE_BASIS_TAKE_PROFIT = enable_basis_tp
            CONFIG.BASIS_TAKE_PROFIT_THRESHOLD = basis_tp_threshold
            CONFIG.BASIS_DECLINE_DAYS = basis_decline_days
            CONFIG.BASIS_MIN_HOLDING_DAYS = basis_min_holding
            CONFIG.ENABLE_DYNAMIC_POSITION = enable_dynamic_position
            CONFIG.LEVERAGE = leverage
            CONFIG.COMMISSION_RATE = commission_rate
            CONFIG.COMMISSION_PER_CONTRACT = commission_per_contract
            CONFIG.USE_FIXED_COMMISSION = use_fixed_commission
            CONFIG.CONTRACT_SIZE = 5  # PTA期货固定为5吨/手
            
            # 验证杠杆倍数是否符合最低保证金要求
            max_leverage = 1.0 / CONFIG.MIN_MARGIN_RATE
            if leverage > max_leverage:
                st.error(f"❌ 杠杆倍数{leverage:.1f}倍超过最大允许值{max_leverage:.1f}倍（最低保证金比例{CONFIG.MIN_MARGIN_RATE*100:.0f}%）")
                st.stop()
            
            # 加载数据
            df = load_merged_data_with_basis(data_path)
            
            # 根据选择的时间段过滤数据
            use_custom_range_val = st.session_state.get('use_custom_range', False)
            backtest_start_date_val = st.session_state.get('backtest_start_date', None)
            backtest_end_date_val = st.session_state.get('backtest_end_date', None)
            
            if use_custom_range_val and backtest_start_date_val and backtest_end_date_val:
                # 确保日期格式正确
                if isinstance(backtest_start_date_val, pd.Timestamp):
                    start_date = backtest_start_date_val
                else:
                    start_date = pd.to_datetime(backtest_start_date_val)
                
                if isinstance(backtest_end_date_val, pd.Timestamp):
                    end_date = backtest_end_date_val
                else:
                    end_date = pd.to_datetime(backtest_end_date_val)
                
                # 过滤数据
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
                
                if len(df) == 0:
                    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                    st.error(f"❌ 在 {start_str} 至 {end_str} 范围内没有数据")
                    st.stop()
                
                start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
                end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                st.info(f"📊 已筛选数据：{len(df)} 条记录（{start_str} 至 {end_str}）")
            
            # 生成交易信号
            df_signals = generate_signals(
                df,
                px_atr_multiplier=px_atr_multiplier,
                margin_long_threshold=margin_long,
                margin_short_threshold=margin_short
            )
            
            # 回测策略（包含杠杆和手续费，根据资金量动态计算手数）
            results = backtest_strategy(
                df_signals,
                initial_capital=initial_capital,
                position_size=position_size,
                max_position_ratio=max_position_ratio,
                holding_period=holding_period,
                atr_multiplier=atr_multiplier,
                basis_take_profit_threshold=basis_tp_threshold,
                leverage=leverage,
                commission_rate=commission_rate,
                commission_per_contract=commission_per_contract,
                use_fixed_commission=use_fixed_commission,
                contract_size=5  # PTA期货固定为5吨/手
            )
            
            # 保存结果到session state
            st.session_state['df'] = df
            st.session_state['df_signals'] = df_signals
            st.session_state['results'] = results
            # 使用不同的key名称保存回测时使用的参数值，避免与widget的key冲突
            st.session_state['backtest_px_atr_multiplier'] = px_atr_multiplier
            st.session_state['backtest_initial_capital'] = initial_capital
            
            st.success("✅ 回测完成！")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 回测失败: {str(e)}")
            st.exception(e)
            st.stop()

# ============================================================================
# 显示回测结果
# ============================================================================
if 'results' in st.session_state:
    results = st.session_state['results']
    df_signals = st.session_state['df_signals']
    
    # 从session state获取回测时使用的参数值（如果存在）
    if 'backtest_px_atr_multiplier' in st.session_state:
        px_atr_multiplier = st.session_state['backtest_px_atr_multiplier']
    else:
        # 如果不存在，尝试从widget读取（回测后widget的值可能已被用户修改）
        px_atr_multiplier = st.session_state.get('px_atr_multiplier', CONFIG.PX_ATR_MULTIPLIER)
    
    if 'backtest_initial_capital' in st.session_state:
        initial_capital = st.session_state['backtest_initial_capital']
    else:
        # 如果不存在，尝试从widget读取
        initial_capital = st.session_state.get('initial_capital', CONFIG.INITIAL_CAPITAL)
    
    # ========== 顶部摘要栏：4个关键指标 ==========
    st.markdown("---")
    st.markdown("## 📊 业绩墙")
    
    # 计算年化收益率
    if len(df_signals) > 0:
        trading_days = len(df_signals)
        years = trading_days / 252
        if years > 0:
            annual_return = ((results['最终资金'] / initial_capital) ** (1/years) - 1) * 100
        else:
            annual_return = 0
    else:
        annual_return = 0
    
    # 计算累计盈利总额
    if len(results['交易记录']) > 0:
        total_profit = sum([t['pnl'] for t in results['交易记录']])
        avg_trade_profit = np.mean([t['pnl'] for t in results['交易记录']])
    else:
        total_profit = 0
        avg_trade_profit = 0
    
    # 4个关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 累计盈利总额",
            f"{total_profit:,.0f} 元",
            delta=f"{results['总收益率']:.2f}%",
            delta_color="normal" if total_profit > 0 else "inverse"
        )
    
    with col2:
        st.metric(
            "📈 年化回报率",
            f"{annual_return:.2f}%",
            help="年化后的收益率，便于对比不同策略"
        )
    
    with col3:
        st.metric(
            "💵 平均每单收益",
            f"{avg_trade_profit:,.0f} 元",
            help="平均每次交易的盈亏金额"
        )
    
    with col4:
        st.metric(
            "🛡️ 历史最大回撤（最稳防线）",
            f"{results['最大回撤']:.2f}%",
            delta="风险指标（越小越好）",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # ========== 资产净值曲线（使用Plotly，完美支持中文） ==========
    st.markdown("## 📈 资产净值曲线")
    
    equity_curve = results['净值曲线']
    dates = df_signals['date'].tolist()[:len(equity_curve)]
    
    # 创建Plotly图表
    fig = go.Figure()
    
    # 绘制净值曲线
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_curve.values,
        mode='lines',
        name='账户净值',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='日期: %{x}<br>账户资金: %{y:,.0f} 元<extra></extra>'
    ))
    
    # 绘制初始资金线
    initial_value = equity_curve.iloc[0]
    fig.add_trace(go.Scatter(
        x=[dates[0], dates[-1]],
        y=[initial_value, initial_value],
        mode='lines',
        name='初始资金',
        line=dict(color='gray', width=2, dash='dash'),
        hovertemplate='初始资金: %{y:,.0f} 元<extra></extra>'
    ))
    
    # 计算并绘制回撤阴影区域
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    drawdown_dates = dates
    drawdown_values = equity_curve.values
    max_values = running_max.values
    
    # 创建回撤区域（填充区域）
    fig.add_trace(go.Scatter(
        x=drawdown_dates + drawdown_dates[::-1],
        y=list(drawdown_values) + list(max_values[::-1]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='回撤区域',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # 标注关键盈利阶段
    if len(results['交易记录']) > 0:
        trades_df = pd.DataFrame(results['交易记录'])
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        
        # 找出盈利最大的交易
        profitable_trades = trades_df[trades_df['pnl'] > 0].sort_values('pnl', ascending=False)
        if len(profitable_trades) > 0:
            top_trade = profitable_trades.iloc[0]
            entry_date = top_trade['entry_date']
            exit_date = top_trade['exit_date']
            
            # 找到对应的净值
            entry_idx = df_signals[df_signals['date'] == entry_date].index
            exit_idx = df_signals[df_signals['date'] == exit_date].index
            
            if len(entry_idx) > 0 and len(exit_idx) > 0:
                exit_equity = equity_curve.iloc[exit_idx[0]]
                annotation_y = exit_equity + (equity_curve.max() - equity_curve.min()) * 0.1
                
                # 添加标注
                fig.add_annotation(
                    x=exit_date,
                    y=exit_equity,
                    ax=exit_date,
                    ay=annotation_y,
                    xref="x",
                    yref="y",
                    text=f'最大盈利单：{top_trade["pnl"]:,.0f}元<br>({entry_date.strftime("%Y-%m")})',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='green',
                    bgcolor='yellow',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10, color='black')
                )
    
    # 标注盈利阶段说明
    if len(df_signals) > 0 and len(results['交易记录']) > 0:
        # 找出PX价差大幅上涨的时期
        df_signals['px_change'] = df_signals['px_naphtha_spread'].pct_change()
        px_surge_periods = df_signals[df_signals['px_change'] > 0.05]  # PX价差单日涨幅>5%
        
        if len(px_surge_periods) > 0:
            # 找到对应的净值增长阶段
            for idx, row in px_surge_periods.head(3).iterrows():  # 只标注前3个
                if idx < len(equity_curve):
                    date_val = row['date']
                    equity_val = equity_curve.iloc[idx]
                    
                    # 检查这个时期是否盈利
                    period_trades = trades_df[
                        (trades_df['entry_date'] <= date_val) & 
                        (trades_df['exit_date'] >= date_val)
                    ]
                    if len(period_trades) > 0 and period_trades['pnl'].sum() > 0:
                        year = date_val.year
                        annotation_y = equity_val + (equity_curve.max() - equity_curve.min()) * 0.15
                        
                        fig.add_annotation(
                            x=date_val,
                            y=equity_val,
                            ax=date_val,
                            ay=annotation_y,
                            xref="x",
                            yref="y",
                            text=f'该阶段盈利核心：<br>捕捉到了{year}年PX暴涨<br>带来的成本传导红利',
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor='orange',
                            bgcolor='lightblue',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=9, color='black'),
                            align='center'
                        )
    
    # 设置图表布局
    fig.update_layout(
        title={
            'text': '策略资产净值曲线（含回撤阴影）',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        xaxis_title='日期',
        yaxis_title='账户资金（元）',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickformat=',.0f'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========== 逻辑共振分布图 ==========
    if len(results['交易记录']) > 0:
        st.markdown("---")
        st.markdown("## 🎯 逻辑共振分布图")
        st.markdown('**为什么我们要等共振？** 当"低加工费 + PX强信号"同时出现时，胜率显著提升')
        
        # 分析共振情况
        trades_df = pd.DataFrame(results['交易记录'])
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        
        # 合并交易记录和信号数据
        resonance_data = []
        for trade in results['交易记录']:
            entry_date = pd.to_datetime(trade['entry_date'])
            # 查找入场日期对应的信号数据
            matching_rows = df_signals[df_signals['date'] == entry_date]
            
            if len(matching_rows) > 0:
                signal_row = matching_rows.iloc[0]
                margin = signal_row.get('pta_margin', np.nan)
                px_change = signal_row.get('px_daily_change_pct', np.nan)
                
                # 判断是否共振：低加工费(<450) + PX强信号(变动>阈值)
                low_margin = not pd.isna(margin) and margin < 450
                # 使用动态阈值判断PX强信号
                px_atr_pct = signal_row.get('px_atr_pct', np.nan)
                if not pd.isna(px_atr_pct):
                    px_threshold = CONFIG.PX_ATR_MULTIPLIER * px_atr_pct / 100
                else:
                    px_threshold = 1.0
                strong_px = not pd.isna(px_change) and abs(px_change) > px_threshold
                resonance = low_margin and strong_px
                
                resonance_data.append({
                    'resonance': '共振' if resonance else '非共振',
                    'profit': '盈利' if trade['pnl'] > 0 else '亏损',
                    'pnl': trade['pnl']
                })
        
        if len(resonance_data) > 0:
            resonance_df = pd.DataFrame(resonance_data)
            
            # 统计共振情况
            resonance_counts = resonance_df['resonance'].value_counts()
            
            if len(resonance_counts) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 共振胜率对比
                    resonance_groups = resonance_df.groupby('resonance')
                    resonance_stats = []
                    for name, group in resonance_groups:
                        if len(group) > 0:
                            win_rate = (group['profit'] == '盈利').sum() / len(group) * 100
                            resonance_stats.append({'类型': name, '胜率(%)': win_rate, '交易次数': len(group)})
                    
                    if len(resonance_stats) > 0:
                        resonance_stats_df = pd.DataFrame(resonance_stats)
                        resonance_stats_df = resonance_stats_df.sort_values('类型', ascending=False)  # 共振在前
                        
                        # 使用Plotly绘制图表（完美支持中文）
                        fig = go.Figure()
                        
                        # 设置颜色
                        colors_list = ['#28a745' if x > 50 else '#ffc107' for x in resonance_stats_df['胜率(%)']]
                        
                        # 添加柱状图
                        fig.add_trace(go.Bar(
                            x=resonance_stats_df['类型'],
                            y=resonance_stats_df['胜率(%)'],
                            marker_color=colors_list,
                            marker_line_color='black',
                            marker_line_width=2,
                            text=[f'{row["胜率(%)"]:.1f}%<br>({row["交易次数"]}次)' 
                                  for _, row in resonance_stats_df.iterrows()],
                            textposition='outside',
                            textfont=dict(size=11, color='black', family='Arial'),
                            hovertemplate='类型: %{x}<br>胜率: %{y:.1f}%<br>交易次数: %{customdata}次<extra></extra>',
                            customdata=resonance_stats_df['交易次数']
                        ))
                        
                        # 添加50%基准线
                        fig.add_hline(y=50, line_dash="dash", line_color="red", 
                                     annotation_text="50%基准线", 
                                     annotation_position="right",
                                     opacity=0.5)
                        
                        # 设置布局
                        fig.update_layout(
                            title={
                                'text': '共振 vs 非共振 胜率对比',
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 14, 'color': 'black'}
                            },
                            xaxis_title='类型',
                            yaxis_title='胜率 (%)',
                            yaxis=dict(range=[0, max(100, resonance_stats_df['胜率(%)'].max() * 1.2)]),
                            height=400,
                            template='plotly_white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("暂无共振数据")
                
                with col2:
                    # 共振平均收益对比
                    resonance_pnl_stats = resonance_df.groupby('resonance')['pnl'].agg(['mean', 'count']).reset_index()
                    resonance_pnl_stats.columns = ['类型', '平均盈亏(元)', '交易次数']
                    resonance_pnl_stats = resonance_pnl_stats.sort_values('类型', ascending=False)  # 共振在前
                    
                    if len(resonance_pnl_stats) > 0:
                        # 使用Plotly绘制图表（完美支持中文）
                        fig = go.Figure()
                        
                        # 设置颜色
                        colors_list = ['#28a745' if x > 0 else '#ffc107' for x in resonance_pnl_stats['平均盈亏(元)']]
                        
                        # 添加柱状图
                        fig.add_trace(go.Bar(
                            x=resonance_pnl_stats['类型'],
                            y=resonance_pnl_stats['平均盈亏(元)'],
                            marker_color=colors_list,
                            marker_line_color='black',
                            marker_line_width=2,
                            text=[f'{row["平均盈亏(元)"]:,.0f}元<br>({row["交易次数"]}次)' 
                                  for _, row in resonance_pnl_stats.iterrows()],
                            textposition='outside',
                            textfont=dict(size=11, color='black', family='Arial'),
                            hovertemplate='类型: %{x}<br>平均盈亏: %{y:,.0f}元<br>交易次数: %{customdata}次<extra></extra>',
                            customdata=resonance_pnl_stats['交易次数']
                        ))
                        
                        # 添加0基准线
                        fig.add_hline(y=0, line_color="black", line_width=1)
                        
                        # 设置布局
                        fig.update_layout(
                            title={
                                'text': '共振 vs 非共振 平均收益对比',
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 14, 'color': 'black'}
                            },
                            xaxis_title='类型',
                            yaxis_title='平均盈亏 (元)',
                            height=400,
                            template='plotly_white',
                            showlegend=False,
                            yaxis=dict(
                                showgrid=True,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("暂无共振数据")
    
    # ========== 交易明细（条件格式 + 平仓原因饼图） ==========
    if len(results['交易记录']) > 0:
        st.markdown("---")
        st.markdown("## 📋 交易明细（操作回顾）")
        
        trades_df = pd.DataFrame(results['交易记录'])
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 平仓原因分布")
            
            # 平仓原因统计
            exit_reasons_map = {
                "固定持仓周期": "持仓到期",
                "价格止损": "价格止损",
                "PX价差跌破均线止损": "PX均线止损",
                "PX价差突破均线止损": "PX均线止损",
                "基差止盈": "基差止盈",
                "回测结束强制平仓": "回测结束"
            }
            
            trades_df['exit_reason_zh'] = trades_df['exit_reason'].map(exit_reasons_map).fillna(trades_df['exit_reason'])
            exit_stats = trades_df['exit_reason_zh'].value_counts()
            
            # 获取字体属性（用于matplotlib图表）
            font_prop = get_chinese_font_prop()
            
            # 绘制饼图
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.Set3(range(len(exit_stats)))
            wedges, texts, autotexts = ax.pie(
                exit_stats.values,
                labels=exit_stats.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontproperties': font_prop, 'fontsize': 10}
            )
            
            # 美化文字
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax.set_title('平仓原因占比', fontproperties=font_prop, fontsize=14, fontweight='bold', pad=20)
            
            st.pyplot(fig)
            plt.close()
            
            # 显示统计说明
            st.caption("💡 这能证明我们的策略是有理有据地进出，而不是盲目持仓")
        
        with col1:
            # 交易明细表（条件格式）
            # 检查是否有手数和手续费字段（新版本才有）
            available_cols = trades_df.columns.tolist()
            display_cols = ["entry_date", "exit_date", "type", "entry_price", "exit_price"]
            
            # 添加手数（如果存在）
            if "contracts" in available_cols:
                display_cols.append("contracts")
            
            display_cols.extend(["pnl", "pnl_pct", "holding_days"])
            
            # 添加手续费（如果存在）
            if "commission" in available_cols:
                display_cols.append("commission")
            
            display_cols.append("exit_reason")
            
            display_df = trades_df[display_cols].copy()
            
            # 构建列名映射
            col_names = ["入场日期", "出场日期", "类型", "入场价", "出场价"]
            if "contracts" in display_cols:
                col_names.append("手数")
            col_names.extend(["盈亏(元)", "收益率(%)", "持仓天数"])
            if "commission" in display_cols:
                col_names.append("手续费(元)")
            col_names.append("平仓原因")
            
            display_df.columns = col_names
            
            # 格式化手数（显示为整数）
            if "手数" in display_df.columns:
                def format_contracts(x):
                    try:
                        if pd.notna(x):
                            val = float(x)
                            return int(val) if val > 0 else 0
                        return 0
                    except:
                        return 0
                display_df["手数"] = display_df["手数"].apply(format_contracts)
            
            # 格式化手续费（保留2位小数）
            if "手续费(元)" in display_df.columns:
                display_df["手续费(元)"] = display_df["手续费(元)"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00")
            
            # 替换平仓原因
            display_df["平仓原因"] = display_df["平仓原因"].map(exit_reasons_map).fillna(display_df["平仓原因"])
            
            # 高亮盈利单（收益率>5%）
            def highlight_profitable(row):
                if row['收益率(%)'] > 5:
                    return ['background-color: #d4edda'] * len(row)
                elif row['收益率(%)'] < -5:
                    return ['background-color: #f8d7da'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = display_df.style.apply(highlight_profitable, axis=1)
            
            st.dataframe(styled_df, use_container_width=True, height=500)
            
            st.caption("💡 绿色背景 = 大肉单（收益率>5%），红色背景 = 大亏单（收益率<-5%）")
            
            # 下载按钮
            csv = trades_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 下载交易明细CSV",
                data=csv,
                file_name=f"交易明细_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # ========== 其他详细指标 ==========
    st.markdown("---")
    st.markdown("## 📊 详细绩效指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总交易次数", f"{results['总交易次数']} 次")
        st.metric("胜率", f"{results['胜率']:.2%}")
    
    with col2:
        st.metric("平均赚的钱 / 平均亏的钱（盈亏比）", f"{results['盈亏比']:.2f}")
        st.metric("稳健度（每承担一份风险换来的钱）", f"{results['夏普比率']:.2f}")
    
    with col3:
        if len(results['交易记录']) > 0:
            winning_trades = [t for t in results['交易记录'] if t['pnl'] > 0]
            losing_trades = [t for t in results['交易记录'] if t['pnl'] <= 0]
            st.metric("盈利交易", f"{len(winning_trades)} 次")
            st.metric("亏损交易", f"{len(losing_trades)} 次")
    
    with col4:
        if len(results['交易记录']) > 0:
            avg_holding = np.mean([t['holding_days'] for t in results['交易记录']])
            st.metric("平均持仓天数", f"{avg_holding:.1f} 天")
            st.metric("最终资金", f"{results['最终资金']:,.0f} 元")
    
    # ========== 价格走势图（使用Plotly，完美支持中文） ==========
    st.markdown("---")
    st.markdown("## 📊 价格走势与交易信号")
    
    # 创建Plotly图表
    fig = go.Figure()
    
    # 绘制PTA期货价格线
    fig.add_trace(go.Scatter(
        x=df_signals["date"],
        y=df_signals["futures_price"],
        mode='lines',
        name='PTA期货价格',
        line=dict(color='#1f77b4', width=1.5),
        opacity=0.7,
        hovertemplate='日期: %{x}<br>价格: %{y:,.0f} 元/吨<extra></extra>'
    ))
    
    # 绘制做多信号
    long_signals = df_signals[df_signals["long_signal"] == True]
    if len(long_signals) > 0:
        fig.add_trace(go.Scatter(
            x=long_signals["date"],
            y=long_signals["futures_price"],
            mode='markers',
            name=f'做多信号 ({len(long_signals)}次)',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='red',
                line=dict(width=1, color='black')
            ),
            hovertemplate='日期: %{x}<br>价格: %{y:,.0f} 元/吨<extra></extra>'
        ))
    
    # 绘制做空信号
    short_signals = df_signals[df_signals["short_signal"] == True]
    if len(short_signals) > 0:
        fig.add_trace(go.Scatter(
            x=short_signals["date"],
            y=short_signals["futures_price"],
            mode='markers',
            name=f'做空信号 ({len(short_signals)}次)',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='blue',
                line=dict(width=1, color='black')
            ),
            hovertemplate='日期: %{x}<br>价格: %{y:,.0f} 元/吨<extra></extra>'
        ))
    
    # 设置图表布局
    fig.update_layout(
        title={
            'text': 'PTA期货价格走势与交易信号（⚠️ 使用期货价格，非现货）',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': 'black'}
        },
        xaxis_title='日期',
        yaxis_title='PTA期货价格（元/吨）',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickformat=',.0f'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========== PX价差走势图（使用Plotly，完美支持中文） ==========
    st.markdown("---")
    st.markdown("## 📈 PX原料利润（价差）走势与交易信号")
    
    # 创建Plotly图表
    fig_px = go.Figure()
    
    # 绘制PX价差线
    fig_px.add_trace(go.Scatter(
        x=df_signals["date"],
        y=df_signals["px_naphtha_spread"],
        mode='lines',
        name='PX原料利润（价差）',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='日期: %{x}<br>PX价差: %{y:.2f} 元/吨<extra></extra>'
    ))
    
    # 如果有动态阈值数据，绘制动态阈值线
    if "dynamic_threshold" in df_signals.columns and "px_naphtha_spread" in df_signals.columns:
        px_prev = df_signals["px_naphtha_spread"].shift(1)
        dynamic_threshold = df_signals["dynamic_threshold"]
        
        # 上阈值线（做多信号触发线）
        upper_threshold = px_prev * (1 + dynamic_threshold / 100)
        fig_px.add_trace(go.Scatter(
            x=df_signals["date"],
            y=upper_threshold,
            mode='lines',
            name='做多信号阈值',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.5,
            hovertemplate='日期: %{x}<br>阈值: %{y:.2f} 元/吨<extra></extra>'
        ))
        
        # 下阈值线（做空信号触发线）
        lower_threshold = px_prev * (1 - dynamic_threshold / 100)
        fig_px.add_trace(go.Scatter(
            x=df_signals["date"],
            y=lower_threshold,
            mode='lines',
            name='做空信号阈值',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.5,
            hovertemplate='日期: %{x}<br>阈值: %{y:.2f} 元/吨<extra></extra>'
        ))
    
    # 绘制做多信号点（在PX价差图上）
    if len(long_signals) > 0:
        long_px_values = df_signals[df_signals["long_signal"] == True]["px_naphtha_spread"]
        fig_px.add_trace(go.Scatter(
            x=long_signals["date"],
            y=long_px_values,
            mode='markers',
            name=f'做多信号 ({len(long_signals)}次)',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='red',
                line=dict(width=1, color='black')
            ),
            hovertemplate='日期: %{x}<br>PX价差: %{y:.2f} 元/吨<extra></extra>'
        ))
    
    # 绘制做空信号点（在PX价差图上）
    if len(short_signals) > 0:
        short_px_values = df_signals[df_signals["short_signal"] == True]["px_naphtha_spread"]
        fig_px.add_trace(go.Scatter(
            x=short_signals["date"],
            y=short_px_values,
            mode='markers',
            name=f'做空信号 ({len(short_signals)}次)',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='blue',
                line=dict(width=1, color='black')
            ),
            hovertemplate='日期: %{x}<br>PX价差: %{y:.2f} 元/吨<extra></extra>'
        ))
    
    # 设置图表布局
    fig_px.update_layout(
        title={
            'text': 'PX原料利润（价差）走势与交易信号',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': 'black'}
        },
        xaxis_title='日期',
        yaxis_title='PX原料利润（价差，元/吨）',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickformat=',.0f'
        )
    )
    
    st.plotly_chart(fig_px, use_container_width=True)
    st.caption("💡 PX原料利润（价差）是策略的核心信号源，当价差变动超过动态阈值时触发交易信号")

else:
    st.info("👆 请在上方上传数据文件并点击'开始回测'按钮执行回测")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>PTA期货策略实战逻辑展示终端 | 基于PX原料利润领先效应</div>",
    unsafe_allow_html=True
)
