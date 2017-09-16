"""MC1-P2: Optimize a portfolio."""
import sys
sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import math
import datetime as dt
import copy


def compute_daily_returns(df):
    dr = df.copy(deep=True)
    dr[1:] = (df[1:] / df[:-1].values) - 1.0
    if type(dr) == pd.Series:
        dr.ix[0] = 0
    else:
        dr.ix[0, :] = 0

    return dr


def compute_portfolio_stats(df_prices, allocs=[0.1, 0.2, 0.3, 0.4], rfr=0.0, sf=252.0, sv=1000000, gen_plot=False):
    # Drop the SPY I added
    spy = df_prices['SPY'].copy(deep=True)
    spy = pd.DataFrame(spy, spy.index) #, dtype=dt.datetime) #This is the troubled line
    df_prices = df_prices.drop('SPY', axis=1)

    df_norm = df_prices / df_prices.ix[0]
    df_value = (df_norm * allocs) * sv
    df_port_val = df_value.sum(axis=1)
    df_portolio_norm = df_port_val / df_port_val.ix[0]
    df_port_dr = compute_daily_returns(df_portolio_norm)[1:]

    cr = (df_port_val.ix[-1] / df_port_val.ix[0]) - 1.0
    adr = df_port_dr.mean()
    sddr = df_port_dr.std()

    # Sharpe Ratio
    df_ret_minus_rfr = (df_port_dr - rfr)
    sr = df_ret_minus_rfr.mean() / df_ret_minus_rfr.std()
    sr = sr * math.sqrt(sf)
    ev = df_port_val[-1]

    # Create Plots
    if gen_plot:
        # Normalzied Portfolio returns vs SPY
        df_daily_portfolio = df_value.sum(axis=1)
        df_daily_portfolio = pd.DataFrame(df_daily_portfolio, df_value.index, ['Portfolio'])
        df_compare = df_daily_portfolio.join(spy)
        df_port_norm = df_compare / df_compare.ix[0]
        plot_data(df_port_norm, title='Portfolio vs SPY', ylabel='Return')


    return cr, adr, sddr, sr, ev




def plot_data(df, title="Stock Prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    plt.savefig(title)



def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method='bfill', inplace=True)

    num_symbols = len(syms)
    first_guess = []
    bounds = []
    for i in range(0,num_symbols):
        first_guess.append(1.0/num_symbols)
        bounds.append((0.0, 1.0))
    allocs = np.array(first_guess)
    parameters = dict(df_prices=prices_all)
    constraints = ({'type': 'eq', 'fun': constrain})
    #print "allocs: ", allocs
    #print "parameters: ", parameters
    #print "constraints: ", constraints

    cr, adr, sddr, sr, ev = compute_portfolio_stats(prices_all, allocs=allocs, gen_plot=gen_plot)

    min_result = spo.minimize(compute_portfolio_sddr, allocs, args=(parameters), constraints=constraints, bounds=bounds)
    #min_result = spo.minimize(compute_portfolio_sddr2, allocs)
    #print min_result
    allocs = min_result['x']

    cr, adr, sddr, sr, ev = compute_portfolio_stats(prices_all, allocs=allocs, gen_plot=gen_plot)

    return allocs, cr, adr, sddr, sr



def compute_portfolio_sddr(allocs, df_prices):
    prices = df_prices['df_prices']
    cr, adr, sddr, sr, ev = compute_portfolio_stats(prices, allocs=allocs)
    return sddr

"""
def compute_portfolio_sddr2(allocs):
    cr, adr, sddr, sr, ev = compute_portfolio_stats(global_df, allocs=allocs)
    return sddr
"""

def constrain(x):
    result = np.sum(x) - 1.0
    return result


def str2dt(strng):
    year,month,day = map(int,strng.split('-'))
    return dt.datetime(year,month,day)

def isclose(a, b, rel_tol=1e-06):
    return abs(a-b) <= rel_tol

def got_error(results, inputs, outputs, text):
    print "___________"
    print "Input: ", inputs
    print "Output: ", outputs
    print "My Results: ", results[0], results[3]
    print results[3]
    print outputs["benchmark"]

def compare_results(results, outputs, inputs):
    # allocs, cr, adr, sddr, sr
    #print "___________"
    #print "Input: ", inputs
    #print "Output: ", outputs
    #print "My Results: ", results[0], results[3]

    #print results[3]
    #print outputs["benchmark"]
    if not isclose(results[3], outputs["benchmark"]):
        got_error(results, inputs, outputs, "sddr is incorrect.")
        return

    if not (len(results[0]) == len(outputs["allocs"])):
        got_error(results, inputs, outputs, "Allocs is incorrect.")
        return

    for i in range(len(results[0])):
        if not isclose(results[0][i], outputs["allocs"][i]):
            got_error(results, inputs, outputs, "Allocs is incorrect.")
            return


    print 'Results match! '



def test_code():
    inputs=dict(
        start_date=str2dt('2010-01-01'),
        end_date=str2dt('2010-12-31'),
        symbols=['GOOG', 'AAPL', 'GLD', 'XOM']
    )

    outputs=dict(
        allocs=[ 0.10612267,  0.00777928,  0.54377087,  0.34232718],
        benchmark=0.00828691718086
    )

    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2004-12-01'),
        end_date=str2dt('2006-05-31'),
        symbols=['YHOO', 'XOM', 'GLD', 'HNZ']
    )
    outputs=dict(
        allocs=[ 0.05963382,  0.07476148,  0.31764505,  0.54795966],
        benchmark=0.00700653270334 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )

    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2005-12-01'),
        end_date=str2dt('2006-05-31'),
        symbols=['YHOO', 'HPQ', 'GLD', 'HNZ']
    )
    outputs=dict(
        allocs=[ 0.10913451,  0.19186373,  0.15370123,  0.54530053],
        benchmark=0.00789501806472 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2005-12-01'),
        end_date=str2dt('2007-05-31'),
        symbols=['MSFT', 'HPQ', 'GLD', 'HNZ']
    )
    outputs=dict(
        allocs=[ 0.29292607,  0.10633076,  0.14849462,  0.45224855],
        benchmark=0.00688155185985 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2006-05-31'),
        end_date=str2dt('2007-05-31'),
        symbols=['MSFT', 'AAPL', 'GLD', 'HNZ']
    )
    outputs=dict(
        allocs=[ 0.20500321,  0.05126107,  0.18217495,  0.56156077],
        benchmark=0.00693253248047 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2011-01-01'),
        end_date=str2dt('2011-12-31'),
        symbols=['AAPL', 'GLD', 'GOOG', 'XOM']
    )
    outputs=dict(
        allocs=[ 0.15673037,  0.51724393,  0.12608485,  0.19994085],
        benchmark=0.0096198317644 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


    inputs=dict(
        start_date=str2dt('2010-06-01'),
        end_date=str2dt('2011-06-01'),
        symbols=['AAPL', 'GLD', 'GOOG']
    )
    outputs=dict(
        allocs=[ 0.21737029,  0.66938007,  0.11324964],
        benchmark=0.00799161174614 # BPH: updated from reference solution, Sunday 3 Sep 2017
    )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)

    inputs=dict(
                start_date=str2dt('2004-01-01'),
                end_date=str2dt('2006-01-01'),
                symbols=['AXP', 'HPQ', 'IBM', 'HNZ']
            )
    outputs=dict(
                allocs=[ 0.29856713,  0.03593918,  0.29612935,  0.36936434],
                benchmark = 0.00706292107796 # BPH: updated from reference solution, Sunday 3 Sep 2017
            )
    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"])
    compare_results(results, outputs, inputs)


def test_case2():
    #Start Date: 2008 - 06 - 01, End Date: 2009 - 06 - 01, Symbols: ['IBM', 'X', 'GLD']
    inputs = dict(
        start_date=str2dt('2008-06-01'),
        end_date=str2dt('2009-06-01'),
        symbols=['IBM', 'X', 'GLD']
    )

    results = optimize_portfolio(sd=inputs["start_date"], ed=inputs["end_date"], syms=inputs["symbols"], gen_plot=True)


def main():
    # Turn on interactive mode for matplotlib
    plt.ion()



if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()




