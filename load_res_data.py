

import pandas as pd
from pathlib import Path
from functools import reduce
# pnl数据载入实例

def load_res_data(fp: Path, file_type: str = "parquet"):
    """
    Load backtest result data.
    #
    Parameters
    ----------
    fp: Path
        File path.
    file_type: str
        File type.
    cols: list
        Columns to load.        
    Returns
    -------
    pd.DataFrame
    """
    files = list(fp.glob("*.parquet"))
    files.sort()
    res_ls = [pd.read_parquet(files[i]).rename(columns={"pnl":f"pnl{i}"}) for i in range(len(files))]
    res_df = reduce(lambda x, y: pd.merge(x, y, on=["Date","basket"], how="left"), res_ls)
    pnl_cols = [f"pnl{i}" for i in range(len(files))]
    res_df.loc[:, "pnl"] = res_df.loc[:, pnl_cols].mean(axis=1)
    t_basket = res_df.basket.max()
    t_res = res_df.loc[res_df.basket == t_basket, ["Date", "pnl"]].reset_index(drop=True)
    return t_res

def calc_max_drawdown(pnl, p_col="pnl"):
    """
    Calculate max drawdown.
    #
    Parameters
    ----------
    pnl: pd.DataFrame
        Pnl series.
    Returns
    -------
    pd.DataFrame
    """
    pnl.loc[:, "cum_ret"] = (1+pnl[p_col]).cumprod()
    pnl.loc[:, "cum_max"] = pnl.cum_ret.cummax()
    pnl.loc[:, "drawdown"] = pnl.cum_max - pnl.cum_ret
    pnl.loc[:, "drawdown_pct"] = pnl.drawdown / pnl.cum_max
    max_drawdown = pnl.drawdown_pct.max()
    return max_drawdown

def calc_ann_indicator(pnl):
    """
    Calculate annualized ret and sp ratio
    #
    Parameters
    ----------
    pnl: pd.DataFrame
        Pnl series.
    Returns
    -------
    pd.DataFrame
    """
    p_col = list(set(pnl.columns) - set(["Date"]))[0]
    pnl.loc[:, "yr"] = pnl.Date.str.slice(0,4)
    # 计算年化收益率
    ann_ret = pnl.groupby("yr").apply(lambda x: (1+x[p_col]).prod()** (251/x.shape[0])-1)
    ann_ret.name = "ann_ret"
    # 计算年化波动率
    ann_vol = pnl.groupby("yr").apply(lambda x: x[p_col].std()* (251**0.5))
    ann_sp = ann_ret.div(ann_vol)
    ann_sp.name = "ann_sp"
    # 计算max_drowdown
    mdd = pnl.groupby("yr").apply(calc_max_drawdown, p_col=p_col)
    mdd.name = "mdd"
    ann_indicator = pd.concat([ann_ret, ann_sp, mdd], axis=1)
    tot_ann_ret = ((1+pnl[p_col]).prod())** (251/pnl.shape[0])-1
    tot_ann_sp = tot_ann_ret / (pnl[p_col].std()* (251**0.5))
    tot_mdd = calc_max_drawdown(pnl, p_col=p_col)
    ann_indicator.loc["total", :] = [tot_ann_ret, tot_ann_sp, tot_mdd]
    return ann_indicator.round(3)


if __name__ == "__main__":
    # path
    pnl_fp = "./backtest_result/fp0020_NA50_CMBevenNap_T1SP10_FctCor60MPNLCor_pos_G10_week_split/pnl"
    fp = Path(pnl_fp)
    index_pnl_fp = "./zz1000_pnl.parquet"
    # load pnl data
    t_res = load_res_data(fp)
    index_pnl = pd.read_parquet(index_pnl_fp)
    # 超额收益率
    excess_ret_df = pd.merge(t_res, index_pnl, on="Date", how="left")
    excess_ret_df.eval("excess_ret = pnl - index_pnl", inplace=True)
    excess_ret_df = excess_ret_df.loc[:,["Date", "excess_ret"]]
    # 多头组合年化收益率和夏普比率
    in_ann_indicator = calc_ann_indicator(t_res.loc[t_res.Date < "20230101"])
    out_ann_indicator = calc_ann_indicator(t_res.loc[t_res.Date >= "20230101"])
    # 多头组合超额年化收益率和夏普比率
    in_excess_ann_indicator = calc_ann_indicator(excess_ret_df.loc[excess_ret_df.Date<"20230101"])
    out_excess_ann_indicator = calc_ann_indicator(excess_ret_df.loc[excess_ret_df.Date>="20230101"])
    print(f"样本内：多头：\n{in_ann_indicator} \n" 
          f"超额：\n {in_excess_ann_indicator} \n",
            f"样本外：多头 \n {out_ann_indicator} \n",
            f"超额：\n {out_excess_ann_indicator} \n")
          