"""Analyze a portfolio."""
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
import math

"""
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    # Add code here to properly compute end value
    ev = sv

    return cr, adr, sddr, sr, ev
"""

def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                     allocs=[0.1, 0.2, 0.3, 0.4],
                     sv=1000000, rfr=0.0, sf=252.0, gen_plot=False):
    """
    Where the returned outputs are:
    cr: Cumulative return
    adr: Average period return (if sf == 252 this is daily return)
    sddr: Standard deviation of daily return
    sr: Sharpe ratio
    ev: End value of portfolio

    The input parameters are:
    sd: A datetime object that represents the start date
    ed: A datetime object that represents the end date
    syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    allocs: A list of 2 or more allocations to the stocks, must sum to 1.0
    sv: Start value of the portfolio
    rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array).
    sf: Sampling frequency per year
    gen_plot: If False, do not create any output. If True it is OK to output a plot such as plot.png
    """
    # Adjust risk free rate to a daily risk free rate
    rfr = calc_daily_rfr(rfr)
    # Get the data for the symbols over the time period asked for
    dates = pd.date_range(sd, ed)
    df_prices = get_data(syms, dates)
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method='bfill', inplace=True)
    # Get Statistics
    cr, adr, sddr, sr, ev = compute_portfolio_stats(df_prices, allocs=allocs, rfr=rfr, sf=sf, gen_plot=gen_plot)

    return cr, adr, sddr, sr, ev #cr, adr, sddr, ev


def calc_daily_rfr(rfr):
    #(1 + risk free rate(rfr) ^ (1 / 252 trading days)) - 1
    #rfr_daily = ((1 + rfr) ** (1.0/252.0)) - 1.0
    return rfr


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    # pd.options.display.float_format = '{:.10f}'.format
    dr = df.copy(deep=True)
    # Start at second date and work all the way down
    # Note: daily_ret[t] = (price[t]/price[t-1]) - 1
    # i.e. Day X price (start on second day) / Day X-1 (i.e. day before) Price, then adjust to % by subtracting one.
    #  Top doesn't include first day, Bottom doesn't include last day. That way they match # of days and we don't overrunn array.
    #  Because first day can't be included, set to zero, since first day's returns are always zero by definition
    dr[1:] = (df[1:] / df[:-1].values) - 1.0
    if type(dr) == pd.Series:
        dr.ix[0] = 0
    else:
        dr.ix[0, :] = 0

    return dr


def compute_portfolio_stats(df_prices, allocs=[0.1, 0.2, 0.3, 0.4], rfr=0.0, sf=252.0, sv=1000000, gen_plot=False):
    """
    cr, adr, sddr, sr = \
        compute_portfolio_stats(prices = df_prices, \
        allocs=[0.1,0.2,0.3,0.4],\
        rfr = 0.0, sf = 252.0)
        Note: I added ev as end value because, why the heck didn't they include that since they want it returned
    """

    # Drop the SPY I added
    spy = df_prices['SPY'].copy(deep=True)
    #spy = pd.DataFrame(spy, spy.index, dtype=dt.datetime)
    df_prices = df_prices.drop('SPY', axis=1)


    # Check for mismatching data
    if len(df_prices.columns) != len(allocs):
        raise Exception("Dataframe and allocations do not match")
    if sum(allocs) != 1.0:
        pass
        # raise Exception("Allocations must sum to 1.0")

    # Normalize data
    df_norm = df_prices / df_prices.ix[0]
    # Turn into daily returns
    df_daily = compute_daily_returns(df_prices)
    # Multiply each column by the allocation to the corresponding equity
    # Multiply these normalized allocations by starting value of overall portfolio, to get position values.
    df_value = (df_norm * allocs) * sv
    # Sum each row (i.e. all position values for each day). That is your daily portfolio value.
    # Use axis 1 because that means we are going to sum across the columns *for every row*
    #df_daily_portfolio = df_value.sum(axis=1)
    #df_daily_portfolio = pd.DataFrame(df_daily_portfolio, df_value.index, ['Portfolio'])
    df_port_val = df_value.sum(axis=1)
    # df_port_val = pd.DataFrame(df_port_val, df_value.index, ['Dollars'])
    # Normalize again to make it easy to get stats
    df_portolio_norm = df_port_val / df_port_val.ix[0]
    #df_portolio_norm = df_daily_portfolio / df_daily_portfolio.ix[0]
    # Turn into daily returns
    df_port_dr = compute_daily_returns(df_portolio_norm)[1:]

    # Cumulative Return
    cr = (df_port_val.ix[-1] / df_port_val.ix[0]) - 1.0
    #cr = (df_daily_portfolio.ix[-1] / df_daily_portfolio.ix[0]) - 1.0
    # Average daily return
    adr = df_port_dr.mean()
    # Standard deviation of daily return
    sddr = df_port_dr.std()

    # Sharpe Ratio
    df_ret_minus_rfr = (df_port_dr - rfr)
    sr = df_ret_minus_rfr.mean() / df_ret_minus_rfr.std()
    # Adjust sharpe ratio from frequence given to annual
    sr = sr * math.sqrt(sf)
    alt_sr = ((adr-rfr) / sddr) * math.sqrt(sf)
    #if not( alt_sr - sr < 0.00001):
    #    raise Exception("Sharpe doesn't match!")
    # end value
    ev = df_port_val[-1]
    #ev = df_portolio_norm[-1]

    # Create Plots
    # TODO: For the class, do the specific plots they ask for. I just did everything I was interested in.
    if gen_plot:
        # Show plots
        # Straight Prices
        """
        plot_data(df_prices, title='Stock Prices', ylabel='Price')
        # Normalized to 1.0
        plot_data(df_norm, title='Stock Returns', ylabel='Return')
        # Daily Returns
        plot_data(df_daily, title='Daily Returns', ylabel='Daily Return')
        # Daily Portfolio Value
        plot_data(df_daily_portfolio, title='Portfolio Value', ylabel='Dollars')
        """
        # Normalzied Portfolio returns vs SPY
        df_daily_portfolio = df_value.sum(axis=1)
        df_daily_portfolio = pd.DataFrame(df_daily_portfolio, df_value.index, ['Portfolio'])
        df_compare = df_daily_portfolio.join(spy)
        df_port_norm = df_compare / df_compare.ix[0]
        plot_data(df_port_norm, title='Portfolio vs SPY', ylabel='Return')


    return cr, adr, sddr, sr, ev




def plot_data(df, title="Stock Prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    plt.savefig(title)



def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    """
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    allocations = [0.0, 0.0, 0.0, 1.0]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    start_date = dt.datetime(2010, 6, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252
    """
    start_date = dt.datetime(2005,4,6)
    end_date = dt.datetime(2010,3,25)
    symbols = ['ETN','KSS','NYT','GPS','BMC','TEL']
    allocations = [0.2, 0.1, 0.1, 0.1, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.06
    sample_freq = 252
    """
    
            'start_date': dt.datetime(2005,4,6),
            'end_date': dt.datetime(2010,3,25),
            'symbols': ['ETN','KSS','NYT','GPS','BMC','TEL'],
            'allocations': [0.2, 0.1, 0.1, 0.1, 0.4, 0.1],
            'start_value': 1000000,
            'risk_free_rate': 0.06,
            'sample_freq': 252
            
            'cumulative_return': 0.626452592439,
            'average_daily_return': 0.000534742431109,
            'volatility': 0.0170978584343,
            'sharpe_ratio': -55.2105225742,
            'end_value': 1626452.59244
    """

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,
        syms = symbols,
        allocs = allocations,
        sv = start_val,
        rfr=risk_free_rate,
        sf=sample_freq,
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    print "End Value: ", ev

if __name__ == "__main__":
    test_code()
