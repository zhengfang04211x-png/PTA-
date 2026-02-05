"""
PTA期货交易策略回测系统 - Streamlit网页版
==========================================

基于PX-石脑油价差"领先效应"的PTA期货交易策略回测系统
支持参数调整、实时回测、结果可视化

运行方式：
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from strategy import (
    StrategyConfig, CONFIG,
    load_merged_data_with_basis,
    generate_signals,
    backtest_strategy,
    get_chinese_font_prop
)

warnings.filterwarnings("ignore")

# 页面配置
st.set_page_config(
    page_title="PTA期货交易策略回测系统",
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
        margin-bottom: 2rem;
    }
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<div class="main-header">📈 PTA期货交易策略回测系统</div>', unsafe_allow_html=True)
st.markdown("---")

# 侧边栏 - 参数配置
st.sidebar.header("⚙️ 策略参数配置")

# 数据文件上传
st.sidebar.subheader("📁 数据文件")
uploaded_file = st.sidebar.file_uploader(
    "上传利润指标合并数据CSV文件",
    type=['csv'],
    help="请上传包含 date, futures_price, px_naphtha_spread, pta_margin 列的CSV文件"
)

# 使用上传的数据
if uploaded_file is not None:
    data_path = uploaded_file
    use_uploaded = True
else:
    data_path = None
    use_uploaded = False
    st.sidebar.warning("⚠️ 请上传数据文件")

# 信号生成参数
st.sidebar.subheader("📊 信号生成参数")
px_atr_period = st.sidebar.slider(
    "PX价差ATR周期",
    min_value=5,
    max_value=50,
    value=CONFIG.PX_ATR_PERIOD,
    step=5,
    help="PX价差ATR计算周期（交易日）"
)

px_atr_multiplier = st.sidebar.slider(
    "PX价差ATR倍数",
    min_value=0.5,
    max_value=3.0,
    value=CONFIG.PX_ATR_MULTIPLIER,
    step=0.1,
    help="动态阈值 = ATR倍数 × PX_ATR"
)

# 估值过滤器参数
st.sidebar.subheader("💰 估值过滤器")
enable_margin_filter = st.sidebar.checkbox(
    "启用估值过滤器",
    value=CONFIG.ENABLE_MARGIN_FILTER,
    help="是否启用估值过滤器"
)

if enable_margin_filter:
    margin_long = st.sidebar.number_input(
        "做多估值阈值 (元/吨)",
        min_value=0,
        max_value=1000,
        value=CONFIG.MARGIN_LONG_THRESHOLD,
        step=10,
        help="仅在pta_margin < 此值时执行做多"
    )
    
    margin_short = st.sidebar.number_input(
        "做空估值阈值 (元/吨)",
        min_value=0,
        max_value=1000,
        value=CONFIG.MARGIN_SHORT_THRESHOLD,
        step=10,
        help="仅在pta_margin > 此值时执行做空"
    )
else:
    margin_long = CONFIG.MARGIN_LONG_THRESHOLD
    margin_short = CONFIG.MARGIN_SHORT_THRESHOLD

# 交易执行参数
st.sidebar.subheader("💼 交易执行参数")
initial_capital = st.sidebar.number_input(
    "初始资金 (元)",
    min_value=100000,
    max_value=10000000,
    value=CONFIG.INITIAL_CAPITAL,
    step=100000,
    format="%d"
)

position_size = st.sidebar.slider(
    "仓位比例 (%)",
    min_value=1,
    max_value=100,
    value=int(CONFIG.POSITION_SIZE * 100),
    step=1,
    help="每次交易的仓位比例"
) / 100

holding_period = st.sidebar.slider(
    "固定持仓周期 (交易日)",
    min_value=5,
    max_value=30,
    value=CONFIG.HOLDING_PERIOD,
    step=1
)

# 风险控制参数
st.sidebar.subheader("🛡️ 风险控制参数")
atr_multiplier = st.sidebar.slider(
    "ATR止损倍数",
    min_value=0.5,
    max_value=3.0,
    value=CONFIG.ATR_MULTIPLIER,
    step=0.1
)

atr_period = st.sidebar.slider(
    "ATR计算周期 (交易日)",
    min_value=5,
    max_value=30,
    value=CONFIG.ATR_PERIOD,
    step=1
)

px_reverse_threshold = st.sidebar.slider(
    "PX反向止损阈值 (%)",
    min_value=1.0,
    max_value=10.0,
    value=CONFIG.PX_REVERSE_THRESHOLD,
    step=0.5
)

# 止盈参数
st.sidebar.subheader("🎯 止盈参数")
enable_basis_tp = st.sidebar.checkbox(
    "启用基差止盈",
    value=CONFIG.ENABLE_BASIS_TAKE_PROFIT,
    help="是否启用基差止盈功能"
)

if enable_basis_tp:
    basis_tp_threshold = st.sidebar.slider(
        "基差止盈盈利阈值 (%)",
        min_value=0.5,
        max_value=5.0,
        value=CONFIG.BASIS_TAKE_PROFIT_THRESHOLD,
        step=0.1
    )
    
    basis_decline_days = st.sidebar.slider(
        "基差连续走弱天数",
        min_value=2,
        max_value=7,
        value=CONFIG.BASIS_DECLINE_DAYS,
        step=1
    )
else:
    basis_tp_threshold = CONFIG.BASIS_TAKE_PROFIT_THRESHOLD
    basis_decline_days = CONFIG.BASIS_DECLINE_DAYS

# 主界面
st.header("🚀 策略回测")

col1, col2 = st.columns([3, 1])

with col2:
    run_backtest = st.button("🚀 开始回测", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📝 当前参数")
    st.write(f"- PX ATR周期: {px_atr_period}")
    st.write(f"- PX ATR倍数: {px_atr_multiplier}")
    st.write(f"- 初始资金: {initial_capital:,.0f} 元")
    st.write(f"- 仓位比例: {position_size*100:.1f}%")
    st.write(f"- 持仓周期: {holding_period} 天")
    st.write(f"- ATR止损: {atr_multiplier}×")
    if enable_margin_filter:
        st.write(f"- 估值过滤: 启用")
    if enable_basis_tp:
        st.write(f"- 基差止盈: 启用")

with col1:
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
                CONFIG.HOLDING_PERIOD = holding_period
                CONFIG.ATR_MULTIPLIER = atr_multiplier
                CONFIG.ATR_PERIOD = atr_period
                CONFIG.PX_REVERSE_THRESHOLD = px_reverse_threshold
                CONFIG.ENABLE_BASIS_TAKE_PROFIT = enable_basis_tp
                CONFIG.BASIS_TAKE_PROFIT_THRESHOLD = basis_tp_threshold
                CONFIG.BASIS_DECLINE_DAYS = basis_decline_days
                
                # 加载数据
                df = load_merged_data_with_basis(data_path)
                
                # 生成交易信号
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("正在生成交易信号...")
                progress_bar.progress(20)
                df_signals = generate_signals(
                    df,
                    px_atr_multiplier=px_atr_multiplier,
                    margin_long_threshold=margin_long,
                    margin_short_threshold=margin_short
                )
                
                # 回测策略
                status_text.text("正在执行策略回测...")
                progress_bar.progress(60)
                results = backtest_strategy(
                    df_signals,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    holding_period=holding_period,
                    atr_multiplier=atr_multiplier,
                    px_reverse_threshold=px_reverse_threshold,
                    basis_take_profit_threshold=basis_tp_threshold
                )
                
                progress_bar.progress(100)
                status_text.text("回测完成！")
                
                # 保存结果到session state
                st.session_state['df'] = df
                st.session_state['df_signals'] = df_signals
                st.session_state['results'] = results
                
                st.success("✅ 回测完成！")
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ 回测失败: {str(e)}")
                st.exception(e)
                st.stop()

# 显示回测结果
if 'results' in st.session_state:
    results = st.session_state['results']
    
    st.markdown("---")
    st.header("📊 回测结果总览")
    
    # 核心指标展示
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        return_color = "positive" if results['总收益率'] > 0 else "negative"
        st.metric(
            "总收益率",
            f"{results['总收益率']:.2f}%",
            delta=f"{results['最终资金'] - CONFIG.INITIAL_CAPITAL:,.0f} 元"
        )
    
    with col2:
        st.metric(
            "总交易次数",
            f"{results['总交易次数']}",
            help="回测期间的总交易次数"
        )
    
    with col3:
        win_rate_color = "positive" if results['胜率'] > 0.5 else "negative"
        st.metric(
            "胜率",
            f"{results['胜率']:.2%}",
            help="盈利交易占总交易的比例"
        )
    
    with col4:
        sharpe_color = "positive" if results['夏普比率'] >= CONFIG.SHARPE_TARGET else "negative"
        st.metric(
            "夏普比率",
            f"{results['夏普比率']:.2f}",
            help="风险调整后的收益指标，≥1.0为优秀"
        )
    
    with col5:
        st.metric(
            "最大回撤",
            f"{results['最大回撤']:.2f}%",
            delta="风险指标",
            delta_color="inverse"
        )
    
    # 详细指标
    st.markdown("---")
    st.subheader("📈 详细绩效指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("最终资金", f"{results['最终资金']:,.0f} 元")
        st.metric("盈亏比", f"{results['盈亏比']:.2f}")
    
    with col2:
        if len(results['交易记录']) > 0:
            winning_trades = [t for t in results['交易记录'] if t['pnl'] > 0]
            losing_trades = [t for t in results['交易记录'] if t['pnl'] <= 0]
            st.metric("盈利交易", f"{len(winning_trades)} 次")
            st.metric("亏损交易", f"{len(losing_trades)} 次")
        else:
            st.metric("盈利交易", "0 次")
            st.metric("亏损交易", "0 次")
    
    with col3:
        if len(results['交易记录']) > 0:
            avg_pnl = np.mean([t['pnl'] for t in results['交易记录']])
            avg_holding = np.mean([t['holding_days'] for t in results['交易记录']])
            st.metric("平均盈亏", f"{avg_pnl:,.0f} 元")
            st.metric("平均持仓天数", f"{avg_holding:.1f} 天")
        else:
            st.metric("平均盈亏", "0 元")
            st.metric("平均持仓天数", "0 天")
    
    with col4:
        if len(results['交易记录']) > 0:
            total_pnl = sum([t['pnl'] for t in results['交易记录']])
            max_win = max([t['pnl'] for t in results['交易记录']])
            max_loss = min([t['pnl'] for t in results['交易记录']])
            st.metric("累计盈亏", f"{total_pnl:,.0f} 元")
            st.metric("最大盈利", f"{max_win:,.0f} 元")
            st.metric("最大亏损", f"{max_loss:,.0f} 元")
        else:
            st.metric("累计盈亏", "0 元")
    
    # 净值曲线
    st.markdown("---")
    st.subheader("📈 净值曲线")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    font_prop = get_chinese_font_prop()
    
    equity_curve = results['净值曲线']
    ax.plot(range(len(equity_curve)), equity_curve, 
            color="#1f77b4", linewidth=2, label="账户净值")
    ax.axhline(y=equity_curve.iloc[0], color="gray", linestyle="--", 
               linewidth=1, alpha=0.5, label="初始资金")
    
    # 标注关键点
    max_equity_idx = equity_curve.idxmax()
    ax.scatter([max_equity_idx], [equity_curve.iloc[max_equity_idx]], 
              color="green", s=100, zorder=5, label="最高净值")
    
    ax.set_xlabel("交易日", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("账户净值（元）", fontproperties=font_prop, fontsize=12)
    ax.set_title("策略净值曲线", fontproperties=font_prop, fontsize=14, fontweight="bold")
    ax.legend(prop=font_prop, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    # 价格走势和交易信号
    st.markdown("---")
    st.subheader("📊 价格走势与交易信号")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_signals = st.session_state['df_signals']
    ax.plot(df_signals["date"], df_signals["futures_price"], 
            color="#1f77b4", linewidth=1.5, alpha=0.7, label="PTA期货价格")
    
    long_signals = df_signals[df_signals["long_signal"] == True]
    short_signals = df_signals[df_signals["short_signal"] == True]
    
    if len(long_signals) > 0:
        ax.scatter(long_signals["date"], long_signals["futures_price"], 
                  color="red", s=100, marker="^", zorder=5, 
                  label=f"做多信号 ({len(long_signals)}次)", edgecolors="black", linewidths=1)
    
    if len(short_signals) > 0:
        ax.scatter(short_signals["date"], short_signals["futures_price"], 
                  color="green", s=100, marker="v", zorder=5, 
                  label=f"做空信号 ({len(short_signals)}次)", edgecolors="black", linewidths=1)
    
    ax.set_xlabel("日期", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel("PTA期货价格（元/吨）", fontproperties=font_prop, fontsize=12)
    ax.set_title("PTA期货价格走势与交易信号", fontproperties=font_prop, fontsize=14, fontweight="bold")
    ax.legend(prop=font_prop, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    plt.close()
    
    # 交易统计
    if len(results['交易记录']) > 0:
        st.markdown("---")
        st.subheader("📋 交易统计分析")
        
        trades_df = pd.DataFrame(results['交易记录'])
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**按交易类型统计**")
            type_stats = trades_df.groupby("type").agg({
                "pnl": ["count", "sum", "mean"],
                "pnl_pct": "mean",
                "holding_days": "mean"
            }).round(2)
            type_stats.columns = ["交易次数", "累计盈亏", "平均盈亏", "平均收益率", "平均持仓天数"]
            st.dataframe(type_stats, use_container_width=True)
        
        with col2:
            st.write("**按平仓原因统计**")
            exit_stats = trades_df.groupby("exit_reason").agg({
                "pnl": ["count", "sum", "mean"]
            }).round(2)
            exit_stats.columns = ["交易次数", "累计盈亏", "平均盈亏"]
            st.dataframe(exit_stats, use_container_width=True)
        
        # 盈亏分布
        st.markdown("---")
        st.subheader("💰 盈亏分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            pnl_values = [t['pnl'] for t in results['交易记录']]
            ax.hist(pnl_values, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="盈亏平衡线")
            ax.set_xlabel("盈亏金额（元）", fontproperties=font_prop, fontsize=11)
            ax.set_ylabel("交易次数", fontproperties=font_prop, fontsize=11)
            ax.set_title("盈亏分布直方图", fontproperties=font_prop, fontsize=12, fontweight="bold")
            ax.legend(prop=font_prop)
            ax.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            pnl_pct_values = [t['pnl_pct'] for t in results['交易记录']]
            ax.hist(pnl_pct_values, bins=20, color="#28a745", alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="盈亏平衡线")
            ax.set_xlabel("收益率 (%)", fontproperties=font_prop, fontsize=11)
            ax.set_ylabel("交易次数", fontproperties=font_prop, fontsize=11)
            ax.set_title("收益率分布直方图", fontproperties=font_prop, fontsize=12, fontweight="bold")
            ax.legend(prop=font_prop)
            ax.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig)
            plt.close()
        
        # 交易明细表
        st.markdown("---")
        st.subheader("📋 交易明细")
        
        # 添加筛选选项
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("筛选交易类型", ["全部", "做多", "做空"])
        with col2:
            filter_profit = st.selectbox("筛选盈亏", ["全部", "盈利", "亏损"])
        with col3:
            sort_by = st.selectbox("排序方式", ["入场日期", "盈亏金额", "收益率", "持仓天数"])
        
        # 应用筛选
        filtered_trades = trades_df.copy()
        if filter_type != "全部":
            filtered_trades = filtered_trades[filtered_trades["type"] == ("long" if filter_type == "做多" else "short")]
        if filter_profit != "全部":
            if filter_profit == "盈利":
                filtered_trades = filtered_trades[filtered_trades["pnl"] > 0]
            else:
                filtered_trades = filtered_trades[filtered_trades["pnl"] <= 0]
        
        # 排序
        if sort_by == "入场日期":
            filtered_trades = filtered_trades.sort_values("entry_date", ascending=False)
        elif sort_by == "盈亏金额":
            filtered_trades = filtered_trades.sort_values("pnl", ascending=False)
        elif sort_by == "收益率":
            filtered_trades = filtered_trades.sort_values("pnl_pct", ascending=False)
        else:
            filtered_trades = filtered_trades.sort_values("holding_days", ascending=False)
        
        # 显示表格
        display_cols = ["entry_date", "exit_date", "type", "entry_price", "exit_price", 
                       "pnl", "pnl_pct", "holding_days", "exit_reason"]
        display_df = filtered_trades[display_cols].copy()
        display_df.columns = ["入场日期", "出场日期", "类型", "入场价", "出场价", 
                              "盈亏(元)", "收益率(%)", "持仓天数", "平仓原因"]
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # 下载按钮
        csv = filtered_trades.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载交易明细CSV",
            data=csv,
            file_name=f"交易明细_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ 没有交易记录，请调整策略参数后重新回测")

else:
    st.info("👆 请在上方配置参数并点击'开始回测'按钮执行回测")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>PTA期货交易策略回测系统 | 基于PX-石脑油价差领先效应</div>",
    unsafe_allow_html=True
)
