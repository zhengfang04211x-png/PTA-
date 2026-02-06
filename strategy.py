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
    POSITION_SIZE = 0.1  # 仓位比例（0-1之间，用于计算投入资金）
    MAX_POSITION_RATIO = 0.8  # 最大仓位比例（最多使用多少比例的资金作为保证金）
    HOLDING_PERIOD = 15  # 固定持仓周期（交易日）
    
    # ========== 期货交易参数 ==========
    LEVERAGE = 1.0  # 杠杆倍数（1.0表示无杠杆，10.0表示10倍杠杆）
    COMMISSION_RATE = 0.0001  # 手续费率（按合约价值的比例，如0.0001表示万分之一，如果使用固定手续费则此参数无效）
    COMMISSION_PER_CONTRACT = 3.3  # 固定手续费（元/手），PTA期货通常为3.3元/手
    USE_FIXED_COMMISSION = True  # 是否使用固定手续费（True=固定金额/手，False=按合约价值比例）
    CONTRACT_SIZE = 5  # 合约单位（PTA期货1手=5吨）
    MIN_MARGIN_RATE = 0.07  # 最低保证金比例（7%，即杠杆最高约14.3倍）
    
    @staticmethod
    def validate_leverage(leverage: float) -> tuple[bool, str]:
        """验证杠杆倍数是否符合最低保证金要求"""
        max_leverage = 1.0 / StrategyConfig.MIN_MARGIN_RATE  # 约14.3倍
        if leverage > max_leverage:
            return False, f"杠杆倍数不能超过{max_leverage:.1f}倍（最低保证金比例{StrategyConfig.MIN_MARGIN_RATE*100:.0f}%）"
        return True, ""
    
    # ========== 风险控制参数 ==========
    ATR_MULTIPLIER = 1.5  # 价格ATR止损倍数
    ATR_PERIOD = 14  # 价格ATR计算周期（交易日）
    PX_MA_PERIOD = 5  # PX价差均线周期（用于动态止损）
    ENABLE_PX_MA_STOP = True  # 是否启用PX均线止损（替代原来的PX反向变动止损）
    
    # ========== 止盈参数 ==========
    BASIS_TAKE_PROFIT_THRESHOLD = 2.0  # 基差止盈盈利阈值（%）
    BASIS_DECLINE_DAYS = 3  # 基差连续走弱天数
    BASIS_MIN_HOLDING_DAYS = 7  # 基差止盈最小持仓天数
    ENABLE_BASIS_TAKE_PROFIT = True  # 是否启用基差止盈
    
    # ========== 分级仓位参数 ==========
    ENABLE_DYNAMIC_POSITION = True  # 是否启用分级仓位
    MARGIN_LOW_THRESHOLD = 350  # 低加工费阈值（仓位放大1.5倍）
    MARGIN_HIGH_THRESHOLD = 600  # 高加工费阈值（仓位缩小至0.5倍）
    POSITION_MULTIPLIER_LOW = 1.5  # 低加工费时的仓位倍数
    POSITION_MULTIPLIER_HIGH = 0.5  # 高加工费时的仓位倍数
    
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
    
    # 期货价格（必须！不是现货价格！）
    # 优先查找：futures_price, 主力合约期货价格, 期货价格
    futures_cols = [c for c in df.columns if "futures" in str(c).lower() and "price" in str(c).lower()]
    if not futures_cols:
        # 查找中文列名：期货价格、主力合约期货价格
        futures_cols = [c for c in df.columns if "期货" in str(c) and "价格" in str(c) and "现货" not in str(c)]
    if not futures_cols:
        # 查找：主力合约
        futures_cols = [c for c in df.columns if "主力合约" in str(c) and "价格" in str(c)]
    if futures_cols:
        result["futures_price"] = pd.to_numeric(df[futures_cols[0]], errors="coerce")
        print(f"[数据加载] 使用期货价格列: {futures_cols[0]}")
    else:
        # 列出所有列名帮助用户排查
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"未找到期货价格列！\n可用列名: {available_cols}\n请确保数据文件包含'futures_price'或包含'期货价格'的列（不是现货价格）")
    
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
        print(f"[数据加载] 从合并数据中找到基差列: {basis_cols[0]}")
    else:
        result["basis"] = np.nan
    
    # 如果合并数据中没有基差，尝试从PTA.csv加载
    if result["basis"].isna().all() and pta_csv_path is not None:
        try:
            pta_path = Path(pta_csv_path)
            if pta_path.exists():
                print(f"[数据加载] 尝试从PTA.csv加载基差数据...")
                pta_df = read_csv_with_encoding(pta_path)
                
                # 识别日期列
                pta_date_cols = [c for c in pta_df.columns if "date" in str(c).lower() or "日期" in str(c) or "时间" in str(c)]
                if not pta_date_cols:
                    for c in pta_df.columns:
                        try:
                            s = pd.to_datetime(pta_df[c], errors="coerce")
                            if s.notna().sum() >= len(pta_df) * 0.5:
                                pta_date_cols = [c]
                                break
                        except:
                            continue
                
                if pta_date_cols:
                    pta_date_col = pta_date_cols[0]
                    pta_df[pta_date_col] = pd.to_datetime(pta_df[pta_date_col], errors="coerce")
                    
                    # 识别现货价格列
                    spot_cols = [c for c in pta_df.columns if "现货" in str(c) and "价格" in str(c)]
                    if not spot_cols:
                        spot_cols = [c for c in pta_df.columns if "spot" in str(c).lower() and "price" in str(c).lower()]
                    
                    # 识别期货价格列（主力合约）
                    pta_futures_cols = [c for c in pta_df.columns if "主力合约" in str(c) and "价格" in str(c)]
                    if not pta_futures_cols:
                        pta_futures_cols = [c for c in pta_df.columns if "期货" in str(c) and "价格" in str(c) and "现货" not in str(c)]
                    
                    # 如果有现货和期货价格，计算基差
                    if spot_cols and pta_futures_cols:
                        pta_basis_df = pd.DataFrame({
                            "date": pta_df[pta_date_col],
                            "spot_price": pd.to_numeric(pta_df[spot_cols[0]], errors="coerce"),
                            "futures_price_pta": pd.to_numeric(pta_df[pta_futures_cols[0]], errors="coerce")
                        })
                        pta_basis_df = pta_basis_df.dropna()
                        # 计算基差：期货价格 - 现货价格
                        pta_basis_df["basis"] = pta_basis_df["futures_price_pta"] - pta_basis_df["spot_price"]
                        
                        # 合并基差数据
                        result = pd.merge_asof(
                            result.sort_values("date"),
                            pta_basis_df[["date", "basis"]].sort_values("date"),
                            on="date",
                            direction="nearest",
                            tolerance=pd.Timedelta(days=7)  # 允许7天内的最近匹配
                        )
                        print(f"[数据加载] ✅ 成功从PTA.csv加载基差数据，有效数据: {result['basis'].notna().sum()} 条")
                    else:
                        # 尝试直接查找基差列（可能是负数）
                        pta_basis_cols = [c for c in pta_df.columns if "基差" in str(c)]
                        if pta_basis_cols:
                            pta_basis_df = pd.DataFrame({
                                "date": pta_df[pta_date_col],
                                "basis": pd.to_numeric(pta_df[pta_basis_cols[0]], errors="coerce")
                            })
                            pta_basis_df = pta_basis_df.dropna()
                            
                            result = pd.merge_asof(
                                result.sort_values("date"),
                                pta_basis_df.sort_values("date"),
                                on="date",
                                direction="nearest",
                                tolerance=pd.Timedelta(days=7)
                            )
                            print(f"[数据加载] ✅ 成功从PTA.csv加载基差列，有效数据: {result['basis'].notna().sum()} 条")
        except Exception as e:
            print(f"[数据加载] ⚠️ 从PTA.csv加载基差数据失败: {e}")
    
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
    """计算期货价格的ATR（使用futures_price，不是现货价格）"""
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
                                max_position_ratio: float = None,
                                holding_period: int = None,
                                atr_multiplier: float = None, 
                                basis_take_profit_threshold: float = None,
                                leverage: float = None,
                                commission_rate: float = None,
                                commission_per_contract: float = None,
                                use_fixed_commission: bool = None,
                                contract_size: int = None) -> dict:
    """回测策略（支持期货杠杆和手续费，根据资金量动态计算手数）"""
    if initial_capital is None:
        initial_capital = CONFIG.INITIAL_CAPITAL
    if position_size is None:
        position_size = CONFIG.POSITION_SIZE
    if max_position_ratio is None:
        max_position_ratio = CONFIG.MAX_POSITION_RATIO
    if holding_period is None:
        holding_period = CONFIG.HOLDING_PERIOD
    if atr_multiplier is None:
        atr_multiplier = CONFIG.ATR_MULTIPLIER
    if basis_take_profit_threshold is None:
        basis_take_profit_threshold = CONFIG.BASIS_TAKE_PROFIT_THRESHOLD
    if leverage is None:
        leverage = CONFIG.LEVERAGE
    if commission_rate is None:
        commission_rate = CONFIG.COMMISSION_RATE
    if commission_per_contract is None:
        commission_per_contract = CONFIG.COMMISSION_PER_CONTRACT
    if use_fixed_commission is None:
        use_fixed_commission = CONFIG.USE_FIXED_COMMISSION
    if contract_size is None:
        contract_size = CONFIG.CONTRACT_SIZE
    
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    df["atr"] = calculate_atr(df, period=CONFIG.ATR_PERIOD)
    df["px_change_pct"] = df["px_naphtha_spread"].pct_change() * 100
    df["basis_change"] = df["basis"].diff()
    
    # 计算PX价差的5日均线（用于动态止损）
    df["px_ma5"] = df["px_naphtha_spread"].rolling(window=CONFIG.PX_MA_PERIOD, min_periods=1).mean()
    
    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    current_position = None
    
    for i in range(len(df)):
        current_date = df.loc[i, "date"]
        current_price = df.loc[i, "futures_price"]  # 期货价格，不是现货！
        current_px = df.loc[i, "px_naphtha_spread"]
        current_px_ma5 = df.loc[i, "px_ma5"]
        current_atr = df.loc[i, "atr"]
        current_basis = df.loc[i, "basis"]
        current_margin = df.loc[i, "pta_margin"] if "pta_margin" in df.columns else np.nan
        
        if current_position is not None:
            holding_days = (current_date - current_position["entry_date"]).days
            
            # 使用保存的手数和保证金（开仓时已计算）
            contracts = current_position.get("contracts", 0)
            invested_margin = current_position.get("invested_margin", 0)
            entry_price = current_position["entry_price"]
            
            if contracts <= 0 or invested_margin <= 0:
                # 如果手数或保证金无效，强制平仓
                current_position = None
                continue
            
            if current_position["type"] == "long":
                price_change = current_price - entry_price
            else:
                price_change = entry_price - current_price
            
            # 计算价格变动带来的盈亏（不考虑手续费，用于判断）
            pnl_from_price = price_change * contracts * contract_size
            # 盈亏百分比（相对于投入的保证金）
            pnl_pct = (pnl_from_price / invested_margin) * 100 if invested_margin > 0 else 0
            
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
                
                # 动态止损：PX价差收盘价跌破5日均线（替代原来的PX反向变动止损）
                if CONFIG.ENABLE_PX_MA_STOP and not pd.isna(current_px_ma5):
                    if current_px < current_px_ma5:
                        stop_loss_triggered = True
                        stop_loss_reason = "PX价差跌破均线止损"
                
                # 基差止盈：持仓超过7天且盈利>2%，基差连续3天走弱
                if (CONFIG.ENABLE_BASIS_TAKE_PROFIT and 
                    holding_days >= CONFIG.BASIS_MIN_HOLDING_DAYS and
                    pnl_pct > basis_take_profit_threshold and
                    not pd.isna(current_basis) and 
                    len(current_position.get("basis_history", [])) >= CONFIG.BASIS_DECLINE_DAYS):
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
                
                # 动态止损：PX价差收盘价突破5日均线（做空时）
                if CONFIG.ENABLE_PX_MA_STOP and not pd.isna(current_px_ma5):
                    if current_px > current_px_ma5:
                        stop_loss_triggered = True
                        stop_loss_reason = "PX价差突破均线止损"
                
                # 基差止盈：持仓超过7天且盈利>2%，基差连续3天走强（做空时基差走强是利空）
                if (CONFIG.ENABLE_BASIS_TAKE_PROFIT and 
                    holding_days >= CONFIG.BASIS_MIN_HOLDING_DAYS and
                    pnl_pct > basis_take_profit_threshold and
                    not pd.isna(current_basis) and 
                    len(current_position.get("basis_history", [])) >= CONFIG.BASIS_DECLINE_DAYS):
                    basis_history = current_position["basis_history"][-CONFIG.BASIS_DECLINE_DAYS:]
                    if len(basis_history) == CONFIG.BASIS_DECLINE_DAYS:
                        basis_rising = all(basis_history[j] > basis_history[j-1] for j in range(1, CONFIG.BASIS_DECLINE_DAYS))
                        if basis_rising:
                            stop_loss_triggered = True
                            stop_loss_reason = "基差止盈"
            
            if holding_days >= holding_period or stop_loss_triggered:
                # 使用上面已经计算好的变量
                # actual_position_size, invested_capital, contracts, entry_price, price_change 等已在上面计算
                
                # 盈亏 = 价格变动 × 合约数量 × 合约单位
                # 注意：这里不需要再乘以leverage，因为合约数量已经通过leverage计算得出
                pnl_from_price = price_change * contracts * contract_size
                
                # 计算手续费（开仓+平仓）
                if use_fixed_commission:
                    # 固定手续费：每手固定金额（开仓+平仓各收一次）
                    total_commission = commission_per_contract * contracts * 2  # 开仓和平仓各收一次
                else:
                    # 比例手续费：按合约价值的一定比例
                    entry_contract_value = entry_price * contracts * contract_size
                    exit_contract_value = current_price * contracts * contract_size
                    entry_commission = entry_contract_value * commission_rate
                    exit_commission = exit_contract_value * commission_rate
                    total_commission = entry_commission + exit_commission
                
                # 最终盈亏 = 价格变动盈亏 - 手续费
                pnl = pnl_from_price - total_commission
                
                # 更新资金
                capital += pnl
                
                # 计算盈亏百分比（相对于投入的保证金）
                pnl_pct = (pnl / invested_margin) * 100 if invested_margin > 0 else 0
                
                exit_reason = "固定持仓周期" if holding_days >= holding_period else stop_loss_reason
                
                trades.append({
                    "entry_date": current_position["entry_date"],
                    "exit_date": current_date,
                    "type": current_position["type"],
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "exit_reason": exit_reason,
                    "contracts": contracts,
                    "commission": total_commission,
                    "leverage": leverage
                })
                current_position = None
        
        if current_position is None:
            if i > 0 and df.loc[i-1, "long_signal"]:
                entry_price = current_price
                entry_px = current_px
                stop_loss_price = entry_price - atr_multiplier * current_atr
                
                # 根据当前资金量计算最多能开多少手
                # 1. 计算可用保证金（最多使用max_position_ratio比例的资金）
                available_margin = capital * max_position_ratio
                
                # 2. 实际投入的保证金 = 可用保证金 × 每次投入比例
                # 注意：position_size已经是0-1之间的比例，不需要再限制
                invested_margin = available_margin * position_size
                
                # 3. 根据保证金和杠杆倍数计算能控制的合约价值
                contract_value = invested_margin * leverage
                
                # 4. 计算能开的手数（向下取整）
                contracts = int(contract_value / (entry_price * contract_size))
                
                # 6. 如果手数为0，说明资金不足，跳过本次开仓
                if contracts <= 0:
                    continue
                
                # 7. 重新计算实际使用的保证金（基于实际手数）
                actual_contract_value = contracts * entry_price * contract_size
                actual_invested_margin = actual_contract_value / leverage
                
                current_position = {
                    "type": "long",
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "entry_px": entry_px,
                    "stop_loss": stop_loss_price,
                    "contracts": contracts,  # 保存实际开仓手数
                    "invested_margin": actual_invested_margin,  # 保存实际投入的保证金
                    "basis_history": [current_basis] if not pd.isna(current_basis) else []
                }
            elif i > 0 and df.loc[i-1, "short_signal"]:
                entry_price = current_price
                entry_px = current_px
                stop_loss_price = entry_price + atr_multiplier * current_atr
                
                # 根据当前资金量计算最多能开多少手（做空逻辑相同）
                available_margin = capital * max_position_ratio
                
                # 实际投入的保证金 = 可用保证金 × 每次投入比例
                # 注意：position_size已经是0-1之间的比例，不需要再限制
                invested_margin = available_margin * position_size
                
                # 根据保证金和杠杆倍数计算能控制的合约价值
                contract_value = invested_margin * leverage
                
                # 计算能开的手数（向下取整）
                contracts = int(contract_value / (entry_price * contract_size))
                
                if contracts <= 0:
                    continue
                
                actual_contract_value = contracts * entry_price * contract_size
                actual_invested_margin = actual_contract_value / leverage
                
                current_position = {
                    "type": "short",
                    "entry_date": current_date,
                    "entry_price": entry_price,
                    "entry_px": entry_px,
                    "stop_loss": stop_loss_price,
                    "contracts": contracts,
                    "invested_margin": actual_invested_margin,
                    "basis_history": [current_basis] if not pd.isna(current_basis) else []
                }
        
        if current_position is not None:
            # 计算未实现盈亏（使用保存的手数和保证金）
            contracts = current_position.get("contracts", 0)
            entry_price = current_position["entry_price"]
            
            if contracts > 0:
                if current_position["type"] == "long":
                    price_change = current_price - entry_price
                else:
                    price_change = entry_price - current_price
                
                unrealized_pnl_from_price = price_change * contracts * contract_size
                # 未实现盈亏暂不考虑手续费（平仓时才扣除）
                unrealized_pnl = unrealized_pnl_from_price
                equity = capital + unrealized_pnl
            else:
                equity = capital
        else:
            equity = capital
        
        equity_curve.append(equity)
    
    if current_position is not None:
        last_price = df.iloc[-1]["futures_price"]
        last_date = df.iloc[-1]["date"]
        holding_days = (last_date - current_position["entry_date"]).days
        
        # 使用保存的手数和保证金
        contracts = current_position.get("contracts", 0)
        invested_margin = current_position.get("invested_margin", 0)
        entry_price = current_position["entry_price"]
        
        if contracts <= 0 or invested_margin <= 0:
            # 如果手数或保证金无效，跳过
            pass
        else:
            if current_position["type"] == "long":
                price_change = last_price - entry_price
            else:
                price_change = entry_price - last_price
            
            pnl_from_price = price_change * contracts * contract_size
            
            # 计算手续费（开仓+平仓）
            if use_fixed_commission:
                # 固定手续费：每手固定金额（开仓+平仓各收一次）
                total_commission = commission_per_contract * contracts * 2  # 开仓和平仓各收一次
            else:
                # 比例手续费：按合约价值的一定比例
                entry_contract_value = entry_price * contracts * contract_size
                exit_contract_value = last_price * contracts * contract_size
                entry_commission = entry_contract_value * commission_rate
                exit_commission = exit_contract_value * commission_rate
                total_commission = entry_commission + exit_commission
            
            pnl = pnl_from_price - total_commission
            capital += pnl
            pnl_pct = (pnl / invested_margin) * 100 if invested_margin > 0 else 0
            
            trades.append({
                "entry_date": current_position["entry_date"],
                "exit_date": last_date,
                "type": current_position["type"],
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "holding_days": holding_days,
                "exit_reason": "回测结束强制平仓",
                "contracts": contracts,
                "commission": total_commission,
                "leverage": leverage
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
        "最大回撤": max_drawdown,  # 显示为"最倒霉时亏了多少"
        "胜率": win_rate,
        "盈亏比": profit_loss_ratio,  # 显示为"平均赚的钱 / 平均亏的钱"
        "夏普比率": sharpe_ratio,  # 显示为"稳健度"
        "最终资金": equity_series.iloc[-1],
        "交易记录": trades,
        "净值曲线": equity_series
    }

