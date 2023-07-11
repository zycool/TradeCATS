# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: test.py
@time: 2023/7/2 19:11
说明:中信交易端（撮合损失交易法）
"""
import os
import time
import pandas as pd
import numpy as np
import datetime
import math
import operator as opr
from strategy_platform import api as cats_api

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 显示所有列
pd.set_option('display.max_columns', None)
# True就是可以换行显示。设置成False的时候不允许换行
pd.set_option('expand_frame_repr', False)

# ===============
# 框架自带变量
# ===============
acct_type = "S0"  # 账户类型和账户
acct = "shutdown"
# acct_type = "SZHTS0"
# acct = "3966778"
start_time = "9:30:00"  # 程序开始运行时间
end_time = "22:00:00"  # 程序结束时间
# ****************
# 自定义变量
# ****************
TARGET_FILE = "D:/Neo/WorkPlace/每日选股结果/2023-07-10.csv"
TARGET_POS_NUM = 50
TRADE_TIME = "13:05"  # 开始交易的时间，默认当天对标的基准交易价对应的时间往后延时1分钟
FINISH_TIME = '14:45'  # 开始进行收尾的时间，指最后开始扫单&检查赎回金额得时间
redeem_money = 20000  # 赎回的金额，无赎回时默认20000
new_order_interval = 0.5  # 每次下单的时间间隔,单位分钟
DEBUG = True  # 是否模拟盘
up_down_limit_set = set()  # 用于存放交易过程中涨跌停得票
order_list = []  # 用于存放订单数据，策略退出时会自动持久化保存
TIMER_TRADE = None
ORDER_STATUS = ["新单(未结)", '部分成交(未结)', '全成(已结)', '部分撤单(已结)', '全撤(已结)', '拒单(已结)']
# ****************
# 建立空的基准信息 df_bench，数据列分别为：股票代码，目标持有数量，基准价,是否停牌,是否涨跌停
# ****************
df_bench = pd.DataFrame(columns=['Symbol', 'sorted_no', 'paused', 'limit', 'target_cap', 'target_vol',
                                 'currentQty', 'enabledQty', 'bidPrice2', 'askPrice2', 'bidVolume1', 'askVolume1'])


# ****************
# 设置参数初始值
# ****************
# cats_api.add_argument('trade_time', str, 0, TRADE_TIME)  # 开始交易的时间
# cats_api.add_argument('new_order_interval', float, 2, new_order_interval)  # 每次下单的时间间隔
# cats_api.add_argument('redeem_money', int, 0, redeem_money)  # 赎回金额

# ****************
# 自定义函数
# ****************
def get_loss_by_time():
    loss_cent = 0.0013  # 这个值表示争取把万13的手续费赚回来
    now_ = datetime.datetime.now()
    the_date = now_.date().isoformat()
    trade_time = datetime.datetime.strptime(str(the_date + ' ' + TRADE_TIME), "%Y-%m-%d %H:%M")
    reb_datetime = datetime.datetime.strptime(str(the_date + ' ' + FINISH_TIME), "%Y-%m-%d %H:%M")
    if "09:30" < TRADE_TIME < "11:30":  # 上午交易
        var_sec = (reb_datetime - trade_time).seconds - 5400
        task_sec = (now_ - trade_time).seconds - 5400
    else:
        var_sec = (reb_datetime - trade_time).seconds
        task_sec = (now_ - trade_time).seconds
    mid_time = reb_datetime + datetime.timedelta(seconds=int(var_sec / 2) * -1)  # 这个时间往后，loss_cent开始变小
    if now_ > mid_time:
        loss_cent = loss_cent - ((now_ - mid_time).seconds * 2) / var_sec / 200
    return loss_cent, mid_time, task_sec / var_sec


def noon_pass():
    """
    判断是否未中午休市时间，即是否是在11：30到1点之间，在此时间段，则跳过定时器中函数
    :return: True or False
    """
    now_ = datetime.datetime.now()
    hour_ = now_.hour
    if 12 <= hour_ < 13 or (hour_ == 11 and now_.minute >= 30):
        return True
    return False


def get_my_logs_dir():
    """在项目根目录下创建自用 my_logs（天为单位）"""
    base_dir = os.path.abspath(os.curdir).replace('\\', '/')
    today = datetime.datetime.today().date().isoformat()
    logs_dir = base_dir + '/my_logs/' + today + '/'
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    return logs_dir


LOGS_DIR = get_my_logs_dir()


def deal_with_paused(code_li):
    """
    输入股票代码列表，按输入顺序返回停牌列表，为停牌列表
    :param code_li: 输入待选择的股票代码列表
    :return: 停牌列表，为停牌列表
    """
    paused_li, not_paused_li = [], []
    for stk in code_li:
        # 获取该票交易时间点时，今日的累计成交量，如果为0，则表示停牌
        data = cats_api.get_today_min1_bar(stk)
        if data.Volume.astype(float).sum() == 0:
            paused_li.append(stk)
        else:
            not_paused_li.append(stk)
    return paused_li, not_paused_li


def deal_with_init(wait_sec=3):
    log.info('*' * 88)
    log.info("第一步：拿到目标持仓，剔除停牌；文件：{}".format(TARGET_FILE[-14:]))
    init_num = int(TARGET_POS_NUM * 1.4)
    targets = pd.read_csv(TARGET_FILE).code.map(lambda x: x[2:] + '.' + x[:2]).tolist()[:init_num]
    log.info('选定候选股票池长度为：{}'.format(len(targets)))

    paused_target, not_paused_target = deal_with_paused(targets)
    if len(paused_target) > 0:
        log.warn('候选股票池中今日停牌的有：{}，将被剔除候选'.format(paused_target))

    log.info("第二步：拿到当前持仓，标识停牌")
    position_list = get_position()
    if not position_list:
        log.warn("持仓信息为 ’空‘，请确认没毛病？")
        log.warn("这里将暂停 {}S 让你想想要不要停掉CATS".format(wait_sec))
        time.sleep(wait_sec)
        log.warn("没停掉，那就是继续咯...")
        log.info('~' * 88)
        return [], [], not_paused_target
    real_hold_pos_list = [pos_obj.symbol for pos_obj in position_list]
    paused_hold, not_paused_hold = deal_with_paused(real_hold_pos_list)
    if len(paused_hold) > 0:
        log.warn('持仓组合中今日停牌的有：{}，这些票不会被交易'.format(paused_hold))
    return paused_hold, real_hold_pos_list, not_paused_target


def get_position(symbol=None):
    """
    :param symbol:默认为 None，表示查询所有标的持仓信息
    :return: 持仓对象list
    """
    position_list = cats_api.query_position(acct_type, acct, symbol=symbol)
    if position_list is None:
        targets = "全部持仓股票" if symbol is None else symbol
        log.error("没查到持仓信息，标的：{}，将返回空列表".format(targets))
        return []
    return position_list


def get_total_asset():
    """
    Return:总资产(=持仓总市值+可用资金+冻结资金),持仓总市值
    """
    account_info = cats_api.query_account(acct_type, acct)
    account_enabledBalance = account_info.enabledBalance  # 查询可用资金
    account_frozenBalance = account_info.frozenBalance  # 查询冻结资金
    position_list = get_position()
    pos_value = np.array([pos.marketValue for pos in position_list]).sum()  # 查询持仓总市值
    return account_enabledBalance + pos_value + account_frozenBalance, pos_value


def save_position_end_time():
    log.info("~~~进行资产&持仓方面统计~~~")
    csv_file = LOGS_DIR + "df_position.csv"
    txt_file = LOGS_DIR + "account.txt"
    info = cats_api.query_account(acct_type, acct)
    currentBalance = info.currentBalance  # 当前余额
    beginBalance = info.beginBalance  # 昨日余额
    enabledBalance = info.enabledBalance  # 可用数
    fetchBalance = info.fetchBalance  # 可取数
    frozenBalance = info.frozenBalance  # 冻结数

    position_list = get_position()
    pos_value = np.array([pos.marketValue for pos in position_list]).sum()  # 查询持仓总市值

    f = open(txt_file, 'w+')
    txt = "当前时间：{}\n当前余额：{}\n昨日余额：{}\n可用金额：{}\n可取金额：{}\n冻结金额：{}\n持仓市值：{}\n总资产：{}\n( = 持仓市值 + 当前余额)".format(
        datetime.datetime.now().isoformat(), currentBalance, beginBalance, enabledBalance, fetchBalance, frozenBalance, pos_value,
        pos_value + currentBalance
    )
    f.writelines(txt)
    f.close()

    variables = {'symbol': "代码", 'stockName': "名称", 'currentQty': "当前余额", 'enabledQty': "可用数", 'costPrice': "成本价",
                 'marketValue': "参考市值", 'frozenQty': "冻结数", 'beginQty': "昨日余额", 'realBuyQty': "今日买入", 'realSellQty': "今日卖出"}
    df = pd.DataFrame([[getattr(pos, j) for j in list(variables.keys())] for pos in position_list], columns=list(variables.values()))
    df.sort_values("参考市值", ascending=False, inplace=True)
    df.to_csv(csv_file, index=False)
    log.info("~~~~已保存统计数据到目录：{}".format(LOGS_DIR))


def trade_begin(hold_list, paused_list, target_list):
    """
    形成当天对标的基准交易价
    查询的为完整的上一分钟的分时数据，比如现在为 9:21:02 秒，则查询的是 9:20 分的分时线，如果在休市后查询，则查询的为最后一分钟的分时线。
    :param target_list:
    :param hold_list:
    :param paused_list:
    :return:DataFrame，里面包含的字段为 Symbol Date Time ClosePrice
    返回的为存在数据的标的，比如查询 10 个标的，可能只有 9 个标的有分时数据，则返回 9 条记录，如果查询的所有的标的不存在或该标的没有数据则返回 None
    每次查询的数量不能超过 200 支标的，如果超过 200 则会报错
    """
    global df_bench

    pkl_file = LOGS_DIR + "df_bench.pkl"
    if os.path.exists(pkl_file):
        log.info("已到达交易时间：{}，今日运行过交易，直接进入  --->>> 运行中  第二阶段".format(TRADE_TIME))
        log.info(pkl_file)
        df_bench = pd.read_pickle(pkl_file)
    else:
        log.info("已到达交易时间：{}，进入  --->>> 运行中  第一阶段  --->>> 关键时间点".format(TRADE_TIME))
        df_bench.loc[paused_list, 'paused'] = 1

        log.info("开始构造目标组合")
        total_cap, pos_cap = get_total_asset()
        log.info("目前总资产：{}，持仓总市值：{}，赎回金额：{}".format(total_cap, pos_cap, redeem_money))
        paused_pos_cap = 0
        if len(paused_list) > 0:
            paused_pos = get_position(symbol=paused_list)
            paused_pos_cap = np.array([pos.marketValue for pos in paused_pos]).sum()
            log.info("目前持仓中'停牌'票总市值：{}".format(paused_pos_cap))
        limit_pos_cap = 0
        limit_list = df_bench[df_bench.limit > 0].index.unique().tolist()
        if len(limit_list) > 0:
            limit_pos = get_position(symbol=limit_list)
            limit_pos_cap = np.array([pos.marketValue for pos in limit_pos]).sum()
            log.info("目前持仓中'涨跌停'票总市值：{}".format(limit_pos_cap))

        trade_symbol_list = list(set(hold_list + target_list))
        df = cats_api.get_today_last_min1_bar(trade_symbol_list)
        df = df[['Symbol', 'Date', 'Time', 'ClosePrice']].copy()
        df['ClosePrice'] = pd.to_numeric(df['ClosePrice'])
        if df.shape[0] < len(trade_symbol_list):
            codes = set(trade_symbol_list) - set(df['Symbol'].unique().tolist())
            error_codes = codes - set(paused_list)
            if len(error_codes) > 0:
                log.error("注意：这些未停牌的票未拉取到基准收盘价！--->>> {}".format(error_codes))
        df.set_index('Symbol', inplace=True)
        df_bench = pd.merge(df, df_bench, how='outer', left_index=True, right_index=True)
        # print(df_bench)

        log.info("开始计算目标组合中各票持仓数量目标")
        target_hold_num = TARGET_POS_NUM - len(paused_list) - len(limit_list)
        try:
            per_stk_cap = (total_cap - paused_pos_cap - limit_pos_cap - redeem_money) / target_hold_num
            log.info("目标组合中各票持仓市值目标：{}".format(per_stk_cap))
            df_bench.loc[target_list, 'target_cap'] = per_stk_cap  # 这里针对候选票票进行目标市值管理
            df_bench.loc[target_list, 'sorted_no'] = range(1, len(target_list) + 1)  # 并对排序打标
            hold_limited_paused = list(set(limit_list + paused_list))
            real_target_positon = target_list[:TARGET_POS_NUM - len(hold_limited_paused)] + hold_limited_paused
            # 这里只对目标持仓剔除停牌（不剔除涨跌停）得票，进行目标持仓数量得计算，方便后面根据这个发订单
            df_bench.loc[list(set(real_target_positon) - set(paused_list)), 'target_vol'] = df_bench.target_cap / df_bench.ClosePrice
            df_bench['target_vol'] = df_bench.target_vol.apply(lambda vol: math.floor(vol / 100) * 100)

            buy_list = list(set(real_target_positon) - set(hold_list))
            sell_list = list(set(hold_list) - set(real_target_positon))
            log.info('买入股票数为：{}，具体：{}'.format(len(buy_list), buy_list))
            log.info('卖出股票数为：{}，具体：{}'.format(len(sell_list), sell_list))
            log.info('要进行rebalance的票数为：{}'.format(TARGET_POS_NUM - len(sell_list)))
            log.info('接着将对 df_bench 持久化为：{}'.format(pkl_file))
            df_bench.to_pickle(pkl_file)
            log.info('已完成下单前的各种骚操作，马上就是见证奇迹的时候啦。。。')
        except ZeroDivisionError:
            log.error("注意：目标持仓量：{}，目前持仓中停牌量-涨跌停量：{} - {}，三者相减已经么得可交易得啦。。。".format(
                TARGET_POS_NUM, len(paused_list), len(limit_list)))
            cats_api.stop_strategy_framework()

    log.info('可参与交易的票DF形状：{}'.format(df_bench.shape))
    log.info('目标组合DF形状：{}'.format(df_bench[df_bench['target_vol'] > 0].shape))

    global TIMER_TRADE
    log.info('~~~~开启交易线程，每隔 {} 分钟执行一次交易条件'.format(new_order_interval))
    trade_running(hold_paused=paused_list)  # 这种定时器启动第一次不运行，所以启动定时器前，可先运行一遍该函数
    TIMER_TRADE = cats_api.minute_timer(new_order_interval, trade_running, hold_paused=paused_list)


def __my_submit_batch(df_to_submit, trade_side=1):
    codes = df_to_submit.index.tolist()
    if trade_side == 1:  # 买入
        cats_api.submit_batch_order([acct_type for _ in range(len(codes))], [acct for _ in range(len(codes))], codes,
                                    [1 for _ in range(len(codes))], [0 for _ in range(len(codes))],
                                    df_to_submit.askPrice2.tolist(), df_to_submit.trade_vol.tolist(),
                                    None, on_batch_order_handler, None)
    elif trade_side == 2:  # 卖出
        cats_api.submit_batch_order([acct_type for _ in range(len(codes))], [acct for _ in range(len(codes))], codes,
                                    [2 for _ in range(len(codes))], [0 for _ in range(len(codes))],
                                    df_to_submit.bidPrice2.tolist(), df_to_submit.trade_vol.tolist(), None, on_batch_order_handler, None)
    else:
        log.error("你传入的 trade_side = {}，不是纯粹的买入卖出操作，请检查！！！！".format(trade_side))


def trade_running(*args, **kwargs):
    global df_bench  # 这里面有可能有基准时刻没涨跌停，后面涨跌停的票，需要动态判断
    if noon_pass():
        return

    loss_cent, mid_time, task_percent = get_loss_by_time()  # 这个值表示要把万13的手续费赚回来
    log.info(" ------------->>>  当前可忍受得交易损失值 = {}".format(loss_cent))

    hold_pos = get_position(symbol=None)
    if not hold_pos:
        log.warn('@@@ 再次提醒：没有查询要任何持仓信息，将当作全新开仓处理。。。')
        df_init = df_bench[df_bench['target_vol'] > 0].copy()
        df_init['per_price'] = df_init.askPrice2 / df_init.ClosePrice
        df_buy_order = df_init[df_init['per_price'] < 1 - loss_cent]
        df_buy_order = df_buy_order[df_buy_order['limit'] < 1]  # 剔除基准时间后涨跌停的票
        if not df_buy_order.empty:
            log.info('发现了可交易机会 --->>>')
            log.info(df_buy_order)
            __my_submit_batch(df_buy_order, trade_side=1)

    else:
        codes = [pos.symbol for pos in hold_pos]
        qtys = [[pos.currentQty, pos.enabledQty] for pos in hold_pos]
        df_bench.loc[codes, ['currentQty', 'enabledQty']] = qtys
        # 以当前这个时间截面数据进行交易判断，且先做卖单，后面才有钱买
        df_sells = df_bench[df_bench['target_vol'] < df_bench['currentQty']].copy()
        df_sells_limit = df_sells[df_sells['limit'] > 0]
        if not df_sells_limit.empty:
            log.info('准备减仓（或换出）的票中有涨跌停的，具体：{}，将不会做卖出操作'.format(df_sells_limit.index.tolist()))
            df_sells = df_sells[df_sells['limit'] < 1].copy()
        if df_sells.empty:
            log.info("***** 卖单已经操作完毕 *****")
        else:
            df_sells['per_price'] = df_sells.bidPrice2 / df_sells.ClosePrice
            df_sells['div_vol'] = df_sells['currentQty'] - df_sells['target_vol']
            df_sells['trade_vol'] = df_sells.apply(lambda row: int(min(row['div_vol'], row['currentQty'], row['bidVolume1'])), axis=1)
            df_sells = df_sells[df_sells['trade_vol'] > 0]  # 有可能遇到trade_vol==0的情况
            df_sells_loss = df_sells[df_sells['per_price'] > 1 + loss_cent].copy()
            if not df_sells_loss.empty:
                log.info('df_sells_loss --->>> 满足最大损失值筛选，对这些票：{}，卖出'.format(df_sells_loss.index.tolist()))
                log.info(df_sells_loss)
                __my_submit_batch(df_sells_loss, trade_side=2)
            if datetime.datetime.now() > mid_time:
                df_sells['div_percent'] = df_sells['div_vol'] / df_sells['currentQty']
                df_sells_time = df_sells[df_sells['div_percent'] > 1 - task_percent].copy()  # 找到时间过半，任务未过半的标的
                if not df_sells_time.empty:
                    log.info('找到时间过半，任务未过半的这些票：{}，卖出'.format(df_sells_time.index.tolist()))
                    log.info(df_sells_time)
                    __my_submit_batch(df_sells_time, trade_side=2)

        df_buys = df_bench[df_bench['target_vol'] > df_bench['currentQty']].copy()
        df_buys_limit = df_buys[df_buys['limit'] > 0]
        if not df_buys_limit.empty:
            log.info('准备加仓（或换入）的票中有涨跌停的，具体：{}，将不会做买入操作'.format(df_buys_limit.index.tolist()))
            df_buys = df_buys[df_buys['limit'] < 1].copy()
        if df_buys.empty:
            log.info("***** 买单已经操作完毕 *****")
        else:
            df_buys['per_price'] = df_buys.askPrice2 / df_buys.ClosePrice
            df_buys['currentQty'] = df_buys.currentQty.apply(lambda v: math.floor(v / 100) * 100)
            df_buys['div_vol'] = df_buys['target_vol'] - df_buys['currentQty']
            df_buys['trade_vol'] = df_buys.apply(lambda row: int(min(row['div_vol'], row['askVolume1'])), axis=1)
            df_buys = df_buys[df_buys['trade_vol'] > 0]  # 有可能遇到trade_vol==0的情况
            df_buys_loss = df_buys[df_buys['per_price'] < 1 - loss_cent].copy()
            df_buys_time = pd.DataFrame()
            if datetime.datetime.now() > mid_time:
                df_buys['div_percent'] = df_buys['div_vol'] / df_buys['target_vol']
                df_buys_time = df_buys[df_buys['div_percent'] > 1 - task_percent].copy()  # 找到时间过半，任务未过半的标的

            currentBalance = cats_api.query_account(acct_type, acct).currentBalance
            if not df_buys_loss.empty:
                need_money = (df_buys_loss['trade_vol'] * df_buys_loss['askPrice2']).sum()
                if currentBalance > need_money:
                    log.info('df_buys_loss --->>> 满足最大损失值筛选，对这些票：{}，以卖1量&卖2价，发买入单'.format(df_buys_loss.index.tolist()))
                    log.info(df_buys_loss)
                    __my_submit_batch(df_buys_loss, trade_side=1)
                else:
                    log.warn('就算满足满足最大损失值，此次买单可用资金不够，买不了 --->>> 可用：{}，需要：{}'.format(currentBalance, need_money))

            if not df_buys_time.empty:
                log.info('找到时间过半，任务未过半的这些票，前5：{}，以买1量&买2价，买进'.format(df_buys_time.index.tolist()[:5]))
                log.info(df_buys_time)
                need_money = (df_buys_time['trade_vol'] * df_buys_time['askPrice2']).sum()
                if currentBalance > need_money:
                    __my_submit_batch(df_buys_time, trade_side=1)
                else:
                    log.warn('哎呀呀，此次买单可用资金不够，买不了 --->>> 可用：{}，需要：{}'.format(currentBalance, need_money))

        if not (df_sells_limit.empty or df_buys_limit.empty):
            global up_down_limit_set
            tmp_set = set(df_sells_limit.index.tolist() + df_buys_limit.index.tolist())
            if not opr.eq(up_down_limit_set, tmp_set):
                log.warn("交易过程中，有新的涨跌停情况发生，故又要重算目标持仓")
                try:
                    paused_list = kwargs.get('hold_paused')
                    df_tmp = df_bench[(df_bench['paused'] < 1) & (df_bench['limit'] < 1)].copy()
                    limit_list = df_bench[df_bench.limit > 0].index.unique().tolist()
                    df_tmp.sort_values('sorted_no', inplace=True)

                    paused_pos_cap = 0
                    if len(paused_list) > 0:
                        paused_pos = get_position(symbol=paused_list)
                        paused_pos_cap = np.array([pos.marketValue for pos in paused_pos]).sum()
                        log.info("目前持仓中'停牌'票总市值：{}".format(paused_pos_cap))
                    limit_pos_cap = 0
                    if len(limit_list) > 0:
                        limit_pos = get_position(symbol=limit_list)
                        limit_pos_cap = np.array([pos.marketValue for pos in limit_pos]).sum()
                        log.info("目前持仓中'涨跌停'票总市值：{}".format(limit_pos_cap))

                    total_cap, pos_cap = get_total_asset()
                    log.info("目前总资产：{}，持仓总市值：{}，赎回金额：{}".format(total_cap, pos_cap, redeem_money))
                    target_hold_num = TARGET_POS_NUM - len(paused_list) - len(limit_list)
                    per_stk_cap = (total_cap - paused_pos_cap - limit_pos_cap - redeem_money) / target_hold_num
                    log.info("更新 --->>> 目标组合中各票持仓市值目标：{}".format(per_stk_cap))
                    var_list = df_tmp.index.tolist()[:target_hold_num]
                    df_bench.loc[var_list, 'target_cap'] = per_stk_cap  # 这里针对目标组合中可交易得票票进行目标市值管理
                    # 这里只对目标持仓剔除停牌（不剔除涨跌停）得票，进行目标持仓数量得计算，方便后面根据这个发订单
                    df_bench.loc[var_list, 'target_vol'] = df_bench.target_cap / df_bench.ClosePrice
                    df_bench['target_vol'] = df_bench.target_vol.apply(lambda vol: math.floor(vol / 100) * 100)

                    up_down_limit_set = tmp_set
                except TypeError:
                    log.error("应该是 paused_list = {} 出了问题".format(kwargs.get('hold_paused')))
                    log.error("up_down_limit_set = {} ".format(up_down_limit_set))
                    log.error("tmp_set = {} ".format(tmp_set))


def trade_finish(*args, **kwargs):
    global TIMER_TRADE, df_bench
    cats_api.cancel_timer(TIMER_TRADE)
    time.sleep(10)
    orders = cats_api.query_order(acct_type, acct)
    log.info("下面是到了收尾时间点，若有还没成交得订单，将被撤回。")
    for i in range(len(orders)):
        if orders[i].status < 2:
            log.info("~~~~ 这个订单将被撤掉 ---->>> 票:{}, 方向:{}, 报单量:{}，状态:{}".format(
                orders[i].symbol, orders[i].side, orders[i].qty, ORDER_STATUS[orders[i].status]))
            cats_api.cancel_order(acct_type, acct, orders[i].orderNo, None, None)
    time.sleep(10)
    hold_pos = get_position(symbol=None)
    while len(hold_pos) == 0:
        log.error("什么鬼，收尾阶段没拿到持仓信息，等10后再试试~~~~~~~~~~~")
        time.sleep(10)
    codes = [pos.symbol for pos in hold_pos]
    qtys = [[pos.currentQty, pos.enabledQty] for pos in hold_pos]
    df_bench.loc[codes, ['currentQty', 'enabledQty']] = qtys
    # 以当前这个时间截面数据进行收尾交易，还是先检查卖单
    df = df_bench[df_bench['limit'] < 1]
    df_sells = df[df['target_vol'] < df['currentQty']].copy()
    while not df_sells.empty:
        log.warn(" !!! 卖单还没有操作完毕的，将对他们以买二价和量做扫单清仓，他们是：")
        log.info(df_sells)
        df_sells['div_vol'] = df_sells['currentQty'] - df_sells['target_vol']
        df_sells['trade_vol'] = df_sells.apply(lambda row: int(min(row['div_vol'], row['enabledQty'], row['bidVolume1'])), axis=1)
        __my_submit_batch(df_sells, trade_side=2)
        time.sleep(15)  # 持仓信息查询有10S的延迟
        sell_list = df_sells.index.tolist()
        hold_pos = get_position(symbol=sell_list)
        while len(hold_pos) == 0:
            log.error("什么鬼，收尾阶段没拿到持仓信息，等5S后再试试~~~~~~~~~~~")
            time.sleep(5)
        codes = [pos.symbol for pos in hold_pos]
        qtys = [[pos.currentQty, pos.enabledQty] for pos in hold_pos]
        df_bench.loc[codes, ['currentQty', 'enabledQty']] = qtys
        df = df_bench[df_bench['limit'] < 1]
        df_sells = df[df['target_vol'] < df['currentQty']].copy()
    log.error("~~~~~~~~~~~~~  有卖出操作的票都已经搞完，下面看看赎回和买入情况  ~~~~~~~~~~~")
    account_info = cats_api.query_account(acct_type, acct)
    currentBalance = account_info.currentBalance  # 查询可用资金
    log.info("目前可用资金：{}，赎回金额：{}".format(currentBalance, redeem_money))
    avb_money = currentBalance - redeem_money
    if avb_money > 10000:
        hold_pos = get_position(symbol=None)
        while len(hold_pos) == 0:
            log.error("什么鬼，收尾阶段没拿到持仓信息，等10后再试试~~~~~~~~~~~")
            time.sleep(10)
        df = df_bench[df_bench['limit'] < 1].copy()
        df['per_price'] = df.askPrice2 / df.ClosePrice
        df['marketValue'] = 0

        codes = [pos.symbol for pos in hold_pos]
        values = [[pos.currentQty, pos.marketValue] for pos in hold_pos]
        df.loc[codes, ['currentQty', 'marketValue']] = values
        df['div_vol'] = df['target_vol'] - df['currentQty']
        # log.info("div_vol ------>>>")
        # log.info(df)
        df['div_vol'] = df.div_vol.apply(lambda v: math.floor(v / 100) * 100)
        df_buys = df[df['div_vol'] > 0].copy()
        if not df_buys.empty:
            log.warn("我去，还有买方目标仓位没到位的~~~~~~~~~~~")
            df_buys['trade_vol'] = ((avb_money / df_buys.size) / df_buys['askPrice2']) // 100
        else:
            df_buys = df.sort_values('per_price').iloc[:2].copy()
            df_buys['trade_vol'] = ((avb_money / df_buys.size) / df_buys['askPrice2']) // 100
        buy_list = df_buys.index.tolist()
        log.info("~~~~~~~~~多余资金：{}，将平均的买入以下几只票：{}~~~~~~~~~~~".format(avb_money, buy_list))
        __my_submit_batch(df_buys, trade_side=1)


# ****************
# 自定义回调接口
# ****************
def on_realmd_handler(realmk_obj, cb_arg):
    """实时行情回调函数"""
    global df_bench
    limit = 0
    if realmk_obj.lastPrice == realmk_obj.upperLimit or realmk_obj.lastPrice == realmk_obj.lowerLimit:
        limit = 1
    df_bench.loc[realmk_obj.symbol, ['limit', 'bidPrice2', 'askPrice2', 'bidVolume1', 'askVolume1']] = [limit,
                                                                                                        realmk_obj.bidPrice2, realmk_obj.askPrice2,
                                                                                                        realmk_obj.bidVolume1, realmk_obj.askVolume1]


def on_batch_order_handler(result, cb_arg):
    '''
    批量下单回调函数
    :param result:
    :param cb_arg:
    :return:
    '''
    if result.rc != "0":
        log.info("批量下单失败, 原因:{}".format(result.resp))
        return
    for i in range(len(result.resp)):
        v = result.resp[i]
        if v.order_no == "":
            log.info("批量下单委托成功，但是有错误信息 :{]".format(v.err_msg))


def on_order_handler(order_obj, cb_arg):
    """
    订单回调函数，打印委托时间，成交数量等信息
    报单状态 status 参数说明:0 新单(未结)，1 部分成交(未结)，2 全成(已结)，3 部分撤单(已结)，4 全撤(已结)，5 拒单(已结)
    """
    global order_list
    if order_obj.status is [1, 3, 4, 5]:
        log.info(
            "订单未成交(或部成): [symbol:{},  委托价:{}, 委托量:{}, 方向:{},状态:{}, 成交数量:{}, 成交均价:{}, 取消数量:{}, 时间:{},日期: {}]".format(
                order_obj.symbol, order_obj.price, order_obj.qty, order_obj.side, ORDER_STATUS[order_obj.status],
                order_obj.filledQty, order_obj.avgPrice, order_obj.cancelQty, order_obj.orderTime, order_obj.orderDate))

    order_list.append((order_obj.symbol, order_obj.orderNo, order_obj.price, order_obj.qty, order_obj.side, order_obj.status,
                       order_obj.filledQty, order_obj.avgPrice, order_obj.cancelQty, order_obj.orderTime, order_obj.orderDate))


def on_cancel_order_notice_handler(cancelNotice, cb_arg):
    """撤单拒单通知回调函数"""
    log.warn("撤单拒单通知回调函数 @@@")
    log.info(cancelNotice)


# ===============
# 需实现的框架接口
# ===============
def on_init(argument_dict):
    global df_bench
    log.info('*' * 88)
    log.info("--->>> 策略入口")
    paused_codes_hold, real_hold_pos_list, targets = deal_with_init(wait_sec=3)
    df_bench['Symbol'] = list(set(targets + real_hold_pos_list))
    df_bench['sorted_no'] = 9999  # 候选票排序序号
    df_bench['paused'] = 0  # 0 未知， -1 未停牌， 1 停牌
    df_bench['limit'] = 0  # 0 未知， -1 未涨跌停， 1 涨跌停
    df_bench['target_cap'] = 0
    df_bench['target_vol'] = 0
    df_bench['currentQty'] = 0
    df_bench.set_index('Symbol', inplace=True)

    log.info("运行中  第一阶段  --->>> 订单订阅")
    cats_api.register_cancel_order_notice_cb(on_cancel_order_notice_handler, None)  # 注册撤单拒单通知回调函数
    cats_api.sub_cancel_order_notice()  # 订阅撤单拒单通知
    cats_api.register_order_cb(on_order_handler, None)  # 注册订单状态回调函数
    cats_api.sub_order(acct_type, acct)  # 订阅指定账户底下的订单

    log.info("运行中  第一阶段  --->>> 行情订阅")
    cats_api.register_realmd_cb(on_realmd_handler, None)  # 注册订阅标的的回调函数
    cats_api.sub_realmd(list(set(targets + real_hold_pos_list)))  # 订阅标的

    now_str = str(datetime.datetime.now().time())
    if now_str < '15:00':
        if now_str > TRADE_TIME:
            trade_begin(hold_list=real_hold_pos_list, paused_list=paused_codes_hold, target_list=targets)
        else:
            log.info("交易时间为：{}，还没到呢，程序将进入等待状态".format(TRADE_TIME))
            cats_api.at_day_timer(TRADE_TIME, trade_begin, hold_list=real_hold_pos_list, paused_list=paused_codes_hold, target_list=targets)

        if now_str > FINISH_TIME:
            trade_finish()
        else:
            log.info("收尾时间为：{}，还没到呢，程序将进入等待状态".format(FINISH_TIME))
            cats_api.at_day_timer(FINISH_TIME, trade_finish)
    else:
        log.info("已经收盘，将执行收盘统计模块，去看交易日志吧！")
        save_position_end_time()
        cats_api.stop_strategy_framework()

    return


def on_fini():
    '''
    策略退出时调用该函数， 如果不实现则默认函数只打印一条日志
    :return: 无返回值，如果失败需要抛出异常
    '''
    global order_list
    pkl_file = LOGS_DIR + "df_order.pkl"

    log.info("~~~~~策略退出中 ......")
    "~~~订单回调: [symbol:{}, 订单号:{}, 委托价:{}, 委托量:{}, 方向:{},状态:{}, 成交数量:{}, 成交均价:{}, 取消数量:{}, 时间:{},日期: {}]"
    df_order = pd.DataFrame(order_list,
                            columns=['symbol', '订单号', '委托价', '委托量', '方向', '状态', '成交数量', '成交均价', '取消数量', '时间', '日期'])

    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
        df_order = pd.concat([df, df_order])

    df_order.to_pickle(pkl_file)
    log.info("~~~~~ 已保存订单数据，完成退出")


def on_update(dict):
    '''
    更新策略时调用该函数；
    如果不实现该函数，则系统提供一个默认的空函数
    start_time和end_time在系统内部默认会处理，用户函数也可以处理
    :param dict: 参数为dict，value见add_argument函数说明
    :return: 需要2个返回值，第一个返回值表示成功或失败，第二个表示原因
       True, "", 成功
       False, "because xxx" 失败
    '''
    log.info("on_update called ......")
    return True, ""
