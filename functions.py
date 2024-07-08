import numpy as np
import pandas as pd
from scipy.stats import t, norm

## Functions

# Slippage
def slippage_increase(cost_to_buy, slippage_abs, slippage_per):

    cost_to_buy_1 = cost_to_buy + slippage_abs
    cost_to_buy_2 = cost_to_buy*(1 + slippage_per/100)

    cost_to_buy = max(cost_to_buy_1, cost_to_buy_2, 0)

    return cost_to_buy

def slippage_decrease(premium, slippage_abs, slippage_per):

    premium_1 = premium - slippage_abs
    premium_2 = premium * (1 - slippage_per/100)

    premium = max(min(premium_1, premium_2),0)

    return premium

# Select volatility type and add to options_df
def add_volatility(options_df, Vol_IV, Vol_RV, Vol_select='IV'):

    # Select volatility type
    if Vol_select == 'IV':
        IV = Vol_IV
    elif Vol_select == 'RV':    
        IV = Vol_RV

    # Rename column heading
    IV = IV.to_frame(name='IV')
    IV.index.name = '[QUOTE_DATE]'

    # Ensure the index of IV is datetime if not already
    IV.index = pd.to_datetime(IV.index)
    options_df['[QUOTE_DATE]'] = pd.to_datetime(options_df['[QUOTE_DATE]'])

    # Merge the IV values into options_df based on 'QUOTE_DATE'
    options_df = options_df.merge(IV, left_on='[QUOTE_DATE]', right_index=True, how='left', suffixes=('_delete', ''))
    if 'IV_delete' in options_df.columns:
        del options_df['IV_delete']

    return options_df

# Select distribution type
def cdf(x, mu, t_df, dist_select):

    x_normalized = x - mu

    if dist_select == 'norm':
        cdf = norm.cdf(x_normalized)
    elif dist_select == 't':
        cdf = t.cdf(x_normalized, t_df)    

    return cdf

# Calculates the expected value percentage and probability of profit for buying and selling call and put options.
def calculate_ev_and_pop(options_df, mu, t_df, dist_select):
    
    # Inputs
    S = options_df['[UNDERLYING_LAST]']
    K = options_df['[STRIKE]']
    T = options_df['[DTE]'] * 1/365
    r = options_df['RFR']
    sigma = options_df['IV']
    P_buy_call = options_df['[C_BID]']
    P_buy_put = options_df['[P_BID]']

    # Compute d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Compute norms
    N_d1 = cdf(d1, mu, t_df, dist_select)
    N_minus_d1 = cdf(-d1, mu, t_df, dist_select)
    N_d2 = cdf(d2, mu, t_df, dist_select)
    N_minus_d2 = cdf(-d2, mu, t_df, dist_select)
    
    # Probability of profit for buying calls and puts
    options_df['buy call POP%'] = N_d2 * 100.0
    options_df['buy put POP%'] = N_minus_d2 * 100.0
    
    # Expected value for buying call and put
    call_value = (S * N_d1 - K * np.exp(-r * T) * N_d2)
    put_value = (K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1)

    # Expected value is 'valued price' minus actual price
    expected_value_call = call_value - P_buy_call
    expected_value_put = put_value - P_buy_put

    options_df['buy call EV%'] = (expected_value_call / P_buy_call) * 100.0
    options_df['buy put EV%'] = (expected_value_put / P_buy_put) * 100.0
    
    return options_df

# Simulate the backtest
def simulate_backtest(options_df, required_probability_per, required_EV_per, allocation_percentage):

    # Filtering
    df_buy_call = options_df[(options_df['buy call POP%'] > required_probability_per) & (options_df['buy call EV%'] > required_EV_per)]
    df_buy_put = options_df[(options_df['buy put POP%'] > required_probability_per) & (options_df['buy put EV%'] > required_EV_per)]

    # Convert to numpy arrays
    dates_call = df_buy_call['[QUOTE_DATE]'].values
    buy_call_perc = df_buy_call['buy call %'].values
    buy_call_ev = df_buy_call['buy call EV%'].values

    dates_put = df_buy_put['[QUOTE_DATE]'].values
    buy_put_perc = df_buy_put['buy put %'].values
    buy_put_ev = df_buy_put['buy put EV%'].values

    # Union of all unique dates
    all_dates = np.unique(np.concatenate((dates_call, dates_put)))

    # Initialize result arrays
    actual_return = np.zeros(len(all_dates))
    expected_value = np.zeros(len(all_dates))

    # Dictionary to keep track of indexes in the original arrays
    call_index = {date: i for i, date in enumerate(dates_call)}
    put_index = {date: i for i, date in enumerate(dates_put)}

    # Calculate actual return and EV%
    for i, date in enumerate(all_dates):
        call_return = buy_call_perc[call_index[date]] if date in call_index else 0
        call_ev = buy_call_ev[call_index[date]] if date in call_index else 0
        
        put_return = buy_put_perc[put_index[date]] if date in put_index else 0
        put_ev = buy_put_ev[put_index[date]] if date in put_index else 0
        
        actual_return[i] = call_return + put_return
        expected_value[i] = call_ev + put_ev

    ## For options buying
    # Sort call data by expected value in descending order
    df_buy_call = df_buy_call.sort_values(by='buy call EV%', ascending=False)

    # Sort put data by expected value in descending order
    df_buy_put = df_buy_put.sort_values(by='buy put EV%', ascending=False)

    # Calculate number of trades (rows) for each date
    df_buy_call_count = df_buy_call.groupby('[QUOTE_DATE]').size()
    df_buy_put_count = df_buy_put.groupby('[QUOTE_DATE]').size()

    # Calculate the mean of 'buy call %' and 'buy put %' for each date separately
    mean_buy_call_perc = df_buy_call.groupby('[QUOTE_DATE]')['buy call %'].mean()
    mean_buy_put_perc = df_buy_put.groupby('[QUOTE_DATE]')['buy put %'].mean()

    # Calculate the mean of 'buy call EV%' and 'buy put EV%' for each date separately
    mean_buy_call_ev = df_buy_call.groupby('[QUOTE_DATE]')['buy call EV%'].mean()
    mean_buy_put_ev = df_buy_put.groupby('[QUOTE_DATE]')['buy put EV%'].mean()

    # Create structured dataframes for calls and puts separately
    df_buy_call = pd.DataFrame({
        '[QUOTE_DATE]': mean_buy_call_perc.index,
        'call return %': mean_buy_call_perc,
        'call EV%': mean_buy_call_ev,
        'call N trades': df_buy_call_count
    }).set_index('[QUOTE_DATE]')

    df_buy_put = pd.DataFrame({
        '[QUOTE_DATE]': mean_buy_put_perc.index,
        'put return %': mean_buy_put_perc,
        'put EV%': mean_buy_put_ev,
        'put N trades': df_buy_put_count
    }).set_index('[QUOTE_DATE]')

    # Combine the structured dataframes for calls and puts
    df_buy_mean = pd.concat([df_buy_call, df_buy_put], axis=1)

    # Replace NaN with 0
    df_buy_mean.fillna(0, inplace=True)

    # Calculate the mean of 'actual return %' and 'EV%'
    df_buy_mean['actual return %'] = 0.5 * (df_buy_mean['call return %'] + df_buy_mean['put return %'])
    df_buy_mean['EV%'] = 0.5 * (df_buy_mean['call EV%'] + df_buy_mean['put EV%'])
    df_buy_mean['N trades'] = df_buy_mean['call N trades'] + df_buy_mean['put N trades']

    # Perform the backtest
    df_buy_mean['Capital growth actual'] = (1 + (allocation_percentage / 100.0) * (df_buy_mean['actual return %'] / 100.0)).cumprod()
    df_buy_mean['EV% mean'] = df_buy_mean['EV%'].expanding(min_periods=1).mean()
    df_buy_mean['actual return % mean'] = df_buy_mean['actual return %'].expanding(min_periods=1).mean()
    df_buy_mean['N trades total'] = df_buy_mean['N trades'].cumsum()
    df_buy_mean['MAPE'] = ((df_buy_mean['actual return % mean'] - df_buy_mean['EV% mean']).abs() / df_buy_mean['EV% mean']) * 100

    return df_buy_mean

def combined_backtest(options_df, Vol_IV, Vol_RV, Vol_select, mu, t_df, dist_select, required_probability_per, required_EV_per, allocation_percentage):

    options_df = add_volatility(options_df, Vol_IV, Vol_RV, Vol_select)
    options_df = calculate_ev_and_pop(options_df, mu, t_df, dist_select)
    df_buy_mean = simulate_backtest(options_df, required_probability_per, required_EV_per, allocation_percentage)

    return df_buy_mean