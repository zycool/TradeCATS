# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: debug_for_init_pos.py
@time: 2023/7/10 20:09
说明:在模拟盘中将头天的仓位建立好，方便测试换仓算法
"""
import os
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
start_time = "09:00:00"  # 程序开始运行时间
end_time = "10:00:00"  # 程序结束时间
# ****************
# 自定义变量
# ****************
TARGET_FILE = "D:/Neo/WorkPlace/每日选股结果/2023-07-07.csv"
TARGET_POS_NUM = 50
TRADE_TIME = "09:31"
new_order_interval = 0.2  # 每次下单的时间间隔,单位分钟
up_down_limit_set = set()  # 用于存放交易过程中涨跌停得票
TIMER_TRADE = None
# ****************
# 建立空的基准信息 df_bench，数据列分别为：股票代码，目标持有数量，基准价,是否停牌,是否涨跌停
# ****************
df_bench = pd.DataFrame(columns=['Symbol', 'sorted_no', 'paused', 'limit', 'target_cap', 'target_vol',
                                 'currentQty', 'enabledQty', 'bidPrice2', 'askPrice2', 'bidVolume1', 'askVolume1'])


# ****************
# 自定义函数
# ****************

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

    return not_paused_target


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


def trade_begin(target_list):
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

    pkl_file = LOGS_DIR + "df_init.pkl"
    if os.path.exists(pkl_file):
        log.error("有仓位初始化呀，请检查：{}".format(pkl_file))
        df_bench = pd.read_pickle(pkl_file)
    else:
        log.info("仓位初始化开始。。。")

        total_cap, pos_cap = get_total_asset()
        log.info("目前总资产：{}，持仓总市值：{}".format(total_cap, pos_cap))

        df = cats_api.get_today_last_min1_bar(target_list)
        df = df[['Symbol', 'Date', 'Time', 'ClosePrice']].copy()
        df['ClosePrice'] = pd.to_numeric(df['ClosePrice'])
        if df.shape[0] < len(target_list):
            codes = set(target_list) - set(df['Symbol'].unique().tolist())
            if len(codes) > 0:
                log.error("注意：这些未停牌的票未拉取到基准收盘价！--->>> {}".format(codes))
        df.set_index('Symbol', inplace=True)
        df_bench = pd.merge(df, df_bench, how='outer', left_index=True, right_index=True)

        log.info("开始计算目标组合中各票持仓数量目标")
        per_stk_cap = total_cap / TARGET_POS_NUM
        log.info("目标组合中各票持仓市值目标：{}".format(per_stk_cap))
        df_bench.loc[target_list, 'target_cap'] = per_stk_cap  # 这里针对候选票票进行目标市值管理
        df_bench.loc[target_list, 'sorted_no'] = range(1, len(target_list) + 1)  # 并对排序打标
        real_target_positon = target_list[:TARGET_POS_NUM]
        # 这里只对目标持仓剔除停牌（不剔除涨跌停）得票，进行目标持仓数量得计算，方便后面根据这个发订单
        df_bench.loc[real_target_positon, 'target_vol'] = df_bench.target_cap / df_bench.ClosePrice
        df_bench['target_vol'] = df_bench.target_vol.apply(lambda vol: math.floor(vol / 100) * 100)

        df_bench.to_pickle(pkl_file)
        log.info('已完成下单前的各种骚操作，马上建仓。。。')

    global TIMER_TRADE
    log.info('~~~~开启交易线程，每隔 {} 分钟执行一次交易条件'.format(new_order_interval))
    TIMER_TRADE = cats_api.minute_timer(new_order_interval, trade_running)


def __my_submit_batch(df_to_submit, trade_side=1):
    codes = df_to_submit.index.tolist()
    if trade_side == 1:
        cats_api.submit_batch_order([acct_type for _ in range(len(codes))], [acct for _ in range(len(codes))], codes,
                                    [1 for _ in range(len(codes))], [0 for _ in range(len(codes))],
                                    df_to_submit.askPrice2.tolist(), df_to_submit.askVolume1.tolist(),
                                    None, on_batch_order_handler, None)
    elif trade_side == 2:
        cats_api.submit_batch_order([acct_type for _ in range(len(codes))], [acct for _ in range(len(codes))], codes,
                                    [2 for _ in range(len(codes))], [0 for _ in range(len(codes))],
                                    df_to_submit.bidPrice2.tolist(), df_to_submit.trade_vol.tolist(), None, on_batch_order_handler, None)
    else:
        log.error("你传入的 trade_side = {}，不是纯粹的买入卖出操作，请检查！！！！".format(trade_side))


def trade_running(*args, **kwargs):
    global df_bench  # 这里面有可能有基准时刻没涨跌停，后面涨跌停的票，需要动态判断

    hold_pos = get_position(symbol=None)
    if not hold_pos:
        log.warn('~~~~~~~~~~~确认没有查询到任何持仓信息，新开仓处理。。。')
        df_init = df_bench[df_bench['target_vol'] > 0].copy()
        df_buy_order = df_init[df_init['limit'] < 1]  # 剔除基准时间后涨跌停的票
        if not df_buy_order.empty:
            __my_submit_batch(df_buy_order, trade_side=1)
    else:
        codes = [pos.symbol for pos in hold_pos]
        qtys = [[pos.currentQty, pos.enabledQty] for pos in hold_pos]
        df_bench.loc[codes, ['currentQty', 'enabledQty']] = qtys
        # 以当前这个时间截面数据进行交易判断，且先做卖单，后面才有钱买

        df_buys = df_bench[df_bench['target_vol'] > df_bench['currentQty']].copy()
        df_buy_order = df_buys[df_buys['limit'] < 1]
        if not df_buy_order.empty:
            __my_submit_batch(df_buy_order, trade_side=1)

        df_buys_limit = df_buys[df_buys['limit'] > 0]
        if not df_buys_limit.empty:
            global up_down_limit_set
            tmp_set = df_buys_limit.index.tolist()
            if not opr.eq(up_down_limit_set, tmp_set):
                log.warn("交易过程中，有新的涨跌停情况发生，故又要重算目标持仓")
                try:
                    df_tmp = df_bench[df_bench['limit'] < 1].copy()
                    limit_list = df_bench[df_bench.limit > 0].index.unique().tolist()
                    df_tmp.sort_values('sorted_no', inplace=True)

                    limit_pos_cap = 0
                    if len(limit_list) > 0:
                        limit_pos = get_position(symbol=limit_list)
                        limit_pos_cap = np.array([pos.marketValue for pos in limit_pos]).sum()
                        log.info("目前持仓中'涨跌停'票总市值：{}".format(limit_pos_cap))

                    total_cap, pos_cap = get_total_asset()
                    log.info("目前总资产：{}，持仓总市值：{}".format(total_cap, pos_cap))
                    target_hold_num = TARGET_POS_NUM - len(limit_list)
                    per_stk_cap = (total_cap - - limit_pos_cap) / target_hold_num
                    log.info("更新 --->>> 目标组合中各票持仓市值目标：{}".format(per_stk_cap))
                    var_list = df_tmp.index.tolist()[:target_hold_num]
                    df_bench.loc[var_list, 'target_cap'] = per_stk_cap  # 这里针对目标组合中可交易得票票进行目标市值管理
                    # 这里只对目标持仓剔除停牌（不剔除涨跌停）得票，进行目标持仓数量得计算，方便后面根据这个发订单
                    df_bench.loc[var_list, 'target_vol'] = df_bench.target_cap / df_bench.ClosePrice
                    df_bench['target_vol'] = df_bench.target_vol.apply(lambda vol: math.floor(vol / 100) * 100)

                    up_down_limit_set = tmp_set
                except TypeError:
                    log.error("up_down_limit_set = {} ".format(up_down_limit_set))
                    log.error("tmp_set = {} ".format(tmp_set))


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
    log.info("~~~订单回调: [symbol:{}, 订单号:{}, 委托价:{}, 委托量:{}, 方向:{},状态:{}, 成交数量:{}, 成交均价:{}, 取消数量:{}, 时间:{},日期: {}]".
             format(order_obj.symbol, order_obj.orderNo, order_obj.price, order_obj.qty, order_obj.side, order_obj.status,
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
    targets = deal_with_init(wait_sec=3)
    df_bench['Symbol'] = targets
    df_bench['sorted_no'] = 9999  # 候选票排序序号
    df_bench['paused'] = 0  # 0 未知， -1 未停牌， 1 停牌
    df_bench['limit'] = 0  # 0 未知， -1 未涨跌停， 1 涨跌停
    df_bench['target_cap'] = 0
    df_bench['target_vol'] = 0
    df_bench['currentQty'] = 0
    df_bench.set_index('Symbol', inplace=True)

    cats_api.register_cancel_order_notice_cb(on_cancel_order_notice_handler, None)  # 注册撤单拒单通知回调函数
    cats_api.sub_cancel_order_notice()  # 订阅撤单拒单通知
    cats_api.register_order_cb(on_order_handler, None)  # 注册订单状态回调函数
    cats_api.sub_order(acct_type, acct)  # 订阅指定账户底下的订单

    cats_api.register_realmd_cb(on_realmd_handler, None)  # 注册订阅标的的回调函数
    cats_api.sub_realmd(targets)  # 订阅标的

    if str(datetime.datetime.now().time()) > TRADE_TIME:
        trade_begin(target_list=targets)
    else:
        log.info("交易时间为：{}，还没到呢，程序将进入等待状态".format(TRADE_TIME))
        cats_api.at_day_timer(TRADE_TIME, trade_begin, target_list=targets)

    return


def on_fini():
    '''
    策略退出时调用该函数， 如果不实现则默认函数只打印一条日志
    :return: 无返回值，如果失败需要抛出异常
    '''

    log.info("~~~~~策略退出中 ......")


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
