"""
PTA期货交易策略核心代码
=======================

包含策略配置、数据加载、信号生成、回测等核心功能
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ============================================================================
# 策略参数配置类
# ============================================================================

class StrategyConfig:
    """策略参数配置类 - 所有可调参数集中在此"""
    
    # ========== 信号生成参数 ==========
    PX_ATR_PERIOD = 20  # PX价差ATR计算周期（交易日）
    PX_ATR_MULTIPLIER = 1.5  # PX价差ATR倍数
    
    # ========== 估值过滤器参数 ==========
    MARGIN_LONG_THRESHOLD = 450  # 做多估值阈值（元/吨）
    MARGIN_SHORT_THRESHOLD = 750  # 做空估值阈值（元/吨）
    ENABLE_MARGIN_FILTER = True  # 是否启用估值过滤器
    
    # ========== 交易执行参数 ==========
    INITIAL_CAPITAL = 1000000  # 初始资金（元）
    POSITION_SIZE = 0.1  # 仓位比例（0-1之间）
    HOLDING_PERIOD = 15  # 固定持仓周期（交易日）
    
    # ========== 风险控制参数 ==========
    ATR_MULTIPLIER = 1.5  # 价格ATR止损倍数
    ATR_PERIOD = 14  # 价格ATR计算周期（交易日）
    PX_REVERSE_THRESHOLD = 3.0  # PX价差反向变动止损阈值（%）
    
    # ========== 止盈参数 ==========
    BASIS_TAKE_PROFIT_THRESHOLD = 2.0  # 基差止盈盈利阈值（%）
    BASIS_DECLINE_DAYS = 3  # 基差连续走弱天数
    ENABLE_BASIS_TAKE_PROFIT = True  # 是否启用基差止盈
    
    # ========== 绩效评估参数 ==========
    SHARPE_TARGET = 1.0  # 夏普比率目标值
    TRADING_DAYS_PER_YEAR = 252  # 每年交易日数

# 创建全局配置实例
CONFIG = StrategyConfig()

# ============================================================================
# 工具函数
# ============================================================================

def get_chinese_font_prop():
    """获取中文字体属性对象"""
    chinese_fonts = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun"]
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            try:
                return font_manager.FontProperties(family=font)
            except:
                pass
    return None


def read_csv_with_encoding(file_path) -> pd.DataFrame:
    """尝试多种编码方式读取CSV文件"""
    encodings = ["utf-8-sig", "gbk", "gb2312", "utf-8", "cp936"]
    
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"无法使用常见编码读取文件: {file_path}")
    else:
        # 如果是文件对象（Streamlit上传的文件）
        for encoding in encodings:
            try:
                file_path.seek(0)  # 重置文件指针
                return pd.read_csv(file_path, encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError("无法使用常见编码读取文件")


def load_merged_data_with_basis(merged_csv_path=None, pta_csv_path=None) -> pd.DataFrame:
    """加载利润指标合并数据，并尝试加载basis数据"""
    if merged_csv_path is None:
        return None
    
    df = read_csv_with_encoding(merged_csv_path)
    
    # 识别日期列
    date_cols = [c for c in df.columns if "date" in str(c).lower() or "日期" in str(c)]
    if not date_cols:
        for c in df.columns:
            try:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().sum() >= len(df) * 0.5:
                    date_cols = [c]
                    break
            except:
                continue
    
    if not date_cols:
        raise ValueError("无法识别日期列")
    
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    result = pd.DataFrame({"date": df[date_col]})
    
    # 期货价格
    futures_cols = [c for c in df.columns if "futures" in str(c).lower() and "price" in str(c).lower()]
    if not futures_cols:
        futures_cols = [c for c in df.columns if "期货" in str(c) and "价格" in str(c)]
    if futures_cols:
        result["futures_price"] = pd.to_numeric(df[futures_cols[0]], errors="coerce")
    else:
        raise ValueError("未找到futures_price列")
    
    # PX-石脑油价差
    px_cols = [c for c in df.columns if "px" in str(c).lower() and "naphtha" in str(c).lower()]
    if not px_cols:
        px_cols = [c for c in df.columns if "PX" in str(c) and ("石脑油" in str(c) or "价差" in str(c))]
    if px_cols:
        result["px_naphtha_spread"] = pd.to_numeric(df[px_cols[0]], errors="coerce")
    else:
        raise ValueError("未找到px_naphtha_spread列")
    
    # PTA加工费
    margin_cols = [c for c in df.columns if "margin" in str(c).lower() or "加工费" in str(c)]
    if margin_cols:
        result["pta_margin"] = pd.to_numeric(df[margin_cols[0]], errors="coerce")
    else:
        result["pta_margin"] = np.nan
    
    # Basis（如果存在）
    basis_cols = [c for c in df.columns if "basis" in str(c).lower() or "基差" in str(c)]
    if basis_cols:
        result["basis"] = pd.to_numeric(df[basis_cols[0]], errors="coerce")
    else:
        result["basis"] = np.nan
    
    result = result.dropna(subset=["date", "futures_price", "px_naphtha_spread"])
    result = result.sort_values("date").reset_index(drop=True)
    
    return result


def calculate_px_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """计算PX价差的ATR"""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    px = df["px_naphtha_spread"]
    tr = abs(px - px.shift(1))
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算期货价格的ATR"""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    high = df["futures_price"]
    low = df["futures_price"]
    close = df["futures_price"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def generate_signals(df: pd.DataFrame, 
                               px_atr_multiplier: float = None,
                               margin_long_threshold: float = None,
                               margin_short_threshold: float = None) -> pd.DataFrame:
    """生成优化后的交易信号"""
    if px_atr_multiplier is None:
        px_atr_multiplier = CONFIG.PX_ATR_MULTIPLIER
    if margin_long_threshold is None:
        margin_long_threshold = CONFIG.MARGIN_LONG_THRESHOLD
    if margin_short_threshold is None:
        margin_short_threshold = CONFIG.MARGIN_SHORT_THRESHOLD
    
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # 计算PX价差的ATR
    df["px_atr"] = calculate_px_atr(df, period=CONFIG.PX_ATR_PERIOD)
    
    # 计算PX价差的单日变动率
    df["px_daily_change_pct"] = df["px_naphtha_spread"].pct_change() * 100
    
    # 计算动态阈值
    df["px_atr_pct"] = (df["px_atr"] / df["px_naphtha_spread"].shift(1)) * 100
    df["dynamic_threshold"] = px_atr_multiplier * df["px_atr_pct"]
    
    # 做多信号
    df["long_signal_raw"] = df["px_daily_change_pct"] > df["dynamic_threshold"]
    if CONFIG.ENABLE_MARGIN_FILTER and df["pta_margin"].notna().any():
        df["long_signal"] = df["long_signal_raw"] & (df["pta_margin"] < margin_long_threshold)
    else:
        df["long_signal"] = df["long_signal_raw"]
    
    # 做空信号
    df["short_signal_raw"] = df["px_daily_change_pct"] < -df["dynamic_threshold"]
    if CONFIG.ENABLE_MARGIN_FILTER and df["pta_margin"].notna().any():
        df["short_signal"] = df["short_signal_raw"] & (df["pta_margin"] > margin_short_threshold)
    else:
        df["short_signal"] = df["short_signal_raw"]
    
    return df


def backtest_strategy(df: pd.DataFrame, 
                                initial_capital: float = None,
                                position_size: float = None, 
                                holding_period: int = None,
                                atr_multiplier: float = None, 
                                px_reverse_threshold: float = None,
                                basis_take_profit_threshold: float = None) -> dict:
    """回测策略"""
    if initial_capital is None:
        initial_capital = CONFIG.INITIAL_CAPITAL
    if position_size is None:
        position_size = CONFIG.POSITION_SIZE
    if holding_period is None:
        holding_period = CONFIG.HOLDING_PERIOD
    if atr_multiplier is None:
        atr_multiplier = CONFIG.ATR_MULTIPLIER
    if px_reverse_threshold is None:
        px_reverse_threshold = CONFIG.PX_REVERSE_THRESHOLD
    if basis_take_profit_threshold is None:
        basis_take_profit_threshold = CONFIG.BASIS_TAKE_PROFIT_THRESHOLD
    
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    df["atr"] = calculate_atr(df, period=CONFIG.ATR_PERIOD)
    df["px_change_pct"] = df["px_naphtha_spread"].pct_change() * 100
    df["basis_change"] = df["basis"].diff()
    
    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    current_position = None
    
    for i in range(len(df)):
        current_date = df.loc[i, "date"]
        current_price = df.loc[i, "futures_price"]
        current_px = df.loc[i, "px_naphtha_spread"]
        current_atr = df.loc[i, "atr"]
        current_basis = df.loc[i, "basis"]
        
        if current_position is not None:
            holding_days = (current_date - current_position["entry_date"]).days
            
            if current_position["type"] == "long":
                pnl_pct = (current_price / current_position["entry_price"] - 1.0) * 100
            else:
                pnl_pct = (current_position["entry_price"] / current_price - 1.0) * 100
            
            if not pd.isna(current_basis):
                if "basis_history" not in current_position:
                    current_position["basis_history"] = []
                current_position["basis_history"].append(current_basis)
            
            stop_loss_triggered = False
            stop_loss_reason = None
            
            if current_position["type"] == "long":
                if current_price < current_position["stop_loss"]:
                    stop_loss_triggered = True
                    stop_loss_reason = "价格止损"
                
                px_reverse = current_position["entry_px"] - current_px
                px_reverse_pct = (px_reverse / current_position["entry_px"]) * 100 if current_position["entry_px"] > 0 else 0
                if px_reverse_pct > px_reverse_threshold:
                    stop_loss_triggered = True
                    stop_loss_reason = "PX反向变动止损"
                
                if (CONFIG.ENABLE_BASIS_TAKE_PROFIT and 
                    not pd.isna(current_basis) and 
                    len(current_position.get("basis_history", [])) >= CONFIG.BASIS_DECLINE_DAYS and
                    pnl_pct > basis_take_profit_threshold):
                    basis_history = current_position["basis_history"][-CONFIG.BASIS_DECLINE_DAYS:]
                    if len(basis_history) == CONFIG.BASIS_DECLINE_DAYS:
                        basis_declining = all(basis_history[j] < basis_history[j-1] for j in range(1, CONFIG.BASIS_DECLINE_DAYS))
                        if basis_declining:
                            stop_loss_triggered = True
                            stop_loss_reason = "基差止盈"
            
            elif current_position["type"] == "short":
                if current_price > current_position["stop_loss"]:
                    stop_loss_triggered = True
                    stop_loss_reason = "价格止损"
                
                px_reverse = current_px - current_position["entry_px"]
                px_reverse_pct = (px_reverse / abs(current_position["entry_px"])) * 100 if current_position["entry_px"] != 0 else 0
                if px_reverse_pct > px_reverse_threshold:
                    stop_loss_triggered = True
                    stop_loss_reason = "PX反向变动止损"
                
                if (CONFIG.ENABLE_BASIS_TAKE_PROFIT and 
                    not pd.isna(current_basis) and 
                    len(current_position.get("basis_history", [])) >= CONFIG.BASIS_DECLINE_DAYS and
                    pnl_pct > basis_take_profit_threshold):
                    basis_history = current_position["basis_history"][-CONFIG.BASIS_DECLINE_DAYS:]
                    if len(basis_history) == CONFIG.BASIS_DECLINE_DAYS:
                        basis_rising = all(basis_history[j] > basis_history[j-1] for j in range(1, CONFIG.BASIS_DECLINE_DAYS))
                        if basis_rising:
                            stop_loss_triggered = True
                            stop_loss_reason = "基差止盈"
            
            if holding_days >= holding_period:
                pnl = capital * position_size * (pnl_pct / 100)
                capital += pnl
                trades.append({
                    "entry_date": current_position["entry_date"],
                    "exit_date": current_date,
                    "type": current_position["type"],
                    "entry_price": current_position["entry_price"],
                    "exit_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "exit_reason": "固定持仓周期"
                })
                current_position = None
            
            elif stop_loss_triggered:
                pnl = capital * position_size * (pnl_pct / 100)
                capital += pnl
                trades.append({
                    "entry_date": current_position["entry_date"],
                    "exit_date": current_date,
                    "type": current_position["type"],
                    "entry_price": current_position["entry_price"],
                    "exit_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "exit_reason": stop_loss_reason
                })
                current_position = None
        
        if current_position is None:
            if i > 0 and df.loc[i-1, "long_signal"]:
                entry_price = current_price
                entry_px = current_px
                stop_loss_price = entry_price - atr_multiplier * current_atr
                current_position = {
                    "type": "long",
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "entry_px": entry_px,
                    "stop_loss": stop_loss_price,
                    "basis_history": [current_basis] if not pd.isna(current_basis) else []
                }
            elif i > 0 and df.loc[i-1, "short_signal"]:
                entry_price = current_price
                entry_px = current_px
                stop_loss_price = entry_price + atr_multiplier * current_atr
                current_position = {
                    "type": "short",
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "entry_px": entry_px,
                    "stop_loss": stop_loss_price,
                    "basis_history": [current_basis] if not pd.isna(current_basis) else []
                }
        
        if current_position is not None:
            if current_position["type"] == "long":
                unrealized_pnl_pct = (current_price / current_position["entry_price"] - 1.0) * 100
            else:
                unrealized_pnl_pct = (current_position["entry_price"] / current_price - 1.0) * 100
            unrealized_pnl = capital * position_size * (unrealized_pnl_pct / 100)
            equity = capital + unrealized_pnl
        else:
            equity = capital
        
        equity_curve.append(equity)
    
    if current_position is not None:
        last_price = df.iloc[-1]["futures_price"]
        last_date = df.iloc[-1]["date"]
        holding_days = (last_date - current_position["entry_date"]).days
        if current_position["type"] == "long":
            pnl_pct = (last_price / current_position["entry_price"] - 1.0) * 100
        else:
            pnl_pct = (current_position["entry_price"] / last_price - 1.0) * 100
        pnl = capital * position_size * (pnl_pct / 100)
        capital += pnl
        trades.append({
            "entry_date": current_position["entry_date"],
            "exit_date": last_date,
            "type": current_position["type"],
            "entry_price": current_position["entry_price"],
            "exit_price": last_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "exit_reason": "回测结束强制平仓"
        })
    
    equity_series = pd.Series(equity_curve)
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1.0) * 100
    
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    if len(trades) > 0:
        winning_trades = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if any(t["pnl"] <= 0 for t in trades) else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    else:
        win_rate = 0
        profit_loss_ratio = 0
    
    if len(equity_series) > 1:
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(CONFIG.TRADING_DAYS_PER_YEAR)
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    return {
        "总交易次数": len(trades),
        "总收益率": total_return,
        "最大回撤": max_drawdown,
        "胜率": win_rate,
        "盈亏比": profit_loss_ratio,
        "夏普比率": sharpe_ratio,
        "最终资金": equity_series.iloc[-1],
        "交易记录": trades,
        "净值曲线": equity_series
    }


