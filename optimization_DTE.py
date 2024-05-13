# Import functions
import pandas as pd
from math import floor, sqrt, inf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
import scipy
from functools import partial

## Optimization functions
## Optimization functions
def find_call_possible_profit(data, var_long, current_price, DTE_min, std_dev, spread_length,\
                              slippage_abs, slippage_per, t_df):

    # Filter rows for the specified start_date and DTE_min
    filtered_data = data[data['[DTE]'] >= DTE_min]
    if filtered_data.empty:
        return None, 0
    
    # Find the row with the minimum [DTE] value in filtered_data
    min_dte_row = filtered_data[filtered_data['[DTE]'] == filtered_data['[DTE]'].min()]
    if min_dte_row.empty:
        return None, 0
    DTE = min_dte_row['[DTE]'].values[0]
    
        # Find the row where [C_DELTA] is closest to delta-long
    strike_desired_long = current_price*(1 + var_long)
    condition_long = lambda row: abs(row['[STRIKE]'] - strike_desired_long)
    closest_long_row = find_closest_row(min_dte_row, condition_long)
    long_price = closest_long_row['[C_BID]'] 
    long_strike = closest_long_row['[STRIKE]']

    # Ensure that closest_long_row is at least 1 index above closest_short_row    
    try:
        closest_short_row = min_dte_row.iloc[min_dte_row.index.get_loc(closest_long_row.name) - int(spread_length)]
        short_price = closest_short_row['[C_ASK]'] 
        short_strike = closest_short_row['[STRIKE]']
    except IndexError:
        return DTE, 0
    
    # Check if short_strike and long_strike exist
    if pd.isnull(short_strike) or pd.isnull(long_strike):
        return DTE, 0  # Short strike or long strike does not exist
    
    # Apply slippage
    short_price = slippage_decrease(short_price, slippage_abs, slippage_per)
    long_price = slippage_increase(long_price, slippage_abs, slippage_per)

    # Check if short_price is less than or equal to long_price
    if short_price <= long_price:
        return DTE, 0
    
    # Calculate figure of merit
    exp_alpha_per = calc_expected_alpha(short_strike,\
                    long_strike, short_price, long_price, \
                        current_price, std_dev, t_df)
    
    return DTE, exp_alpha_per


def find_put_possible_profit(data, var_long, current_price, DTE_min, std_dev, spread_length,\
                             slippage_abs, slippage_per, t_df):

    # Filter rows for the specified start_date and DTE_min
    filtered_data = data[data['[DTE]'] >= DTE_min]
    if filtered_data.empty:
        return None, None
    
    # Find the row with the minimum [DTE] value in filtered_data
    min_dte_row = filtered_data[filtered_data['[DTE]'] == filtered_data['[DTE]'].min()]
    if min_dte_row.empty:
        return None, None
    DTE = min_dte_row['[DTE]'].values[0]
    
    # Find the row where [C_DELTA] is closest to delta-long
    strike_desired_long = current_price*(1 - var_long)
    condition_long = lambda row: abs(row['[STRIKE]'] - strike_desired_long)
    closest_long_row = find_closest_row(min_dte_row, condition_long)
    long_price = closest_long_row['[P_BID]'] 
    long_strike = closest_long_row['[STRIKE]']

    # Ensure that closest_long_row is at least 1 index above closest_short_row    
    try:
        closest_short_row = min_dte_row.iloc[min_dte_row.index.get_loc(closest_long_row.name) + int(spread_length)]
        short_price = closest_short_row['[P_ASK]'] 
        short_strike = closest_short_row['[STRIKE]']
    except IndexError:
        return DTE, 0    

    # Check if short_strike and long_strike exist
    if pd.isnull(short_strike) or pd.isnull(long_strike):
        return DTE, 0  # Short strike or long strike does not exist
    
    # Apply slippage
    short_price = slippage_decrease(short_price, slippage_abs, slippage_per)
    long_price = slippage_increase(long_price, slippage_abs, slippage_per)
    
    # Check if short_price is less than or equal to long_price
    if short_price <= long_price:
        return DTE, 0  # Short price is less than or equal to long price
        
    # Calculate figure of merit
    exp_alpha_per = calc_expected_alpha(short_strike,\
                    long_strike, short_price, long_price, \
                        current_price, std_dev, t_df)
    
    return DTE, exp_alpha_per
        
def find_opt_DTE(filtered_data, current_price, delta_long, VIX, \
                    DTE_min, DTE_max, spread_length, slippage_abs, slippage_per, \
                        t_df):

    DTE_vect = np.array([])
    ROR_vect = np.array([])
    flag = False

    for DTE_min in range(DTE_min, DTE_max+1, 1):

        DTE = int(DTE_min)
        DTE_prev = int(DTE_min)
        while(1):

            # Obtain the variance required
            var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)

            DTE_call, ROR_call = find_call_possible_profit(\
                filtered_data, var_long, current_price, DTE_min, std_dev, spread_length,\
                    slippage_abs, slippage_per, t_df)
            DTE_put, ROR_put = find_put_possible_profit(\
                filtered_data, var_long, current_price, DTE_min, std_dev, spread_length,\
                    slippage_abs, slippage_per, t_df)

            if DTE_call is None or DTE_put is None:
                flag = True
                break
            
            DTE = int(DTE_call)

            if DTE == DTE_prev:
                break
            DTE_prev = DTE

        if flag == True:
            break

        ROR = (ROR_call + ROR_put)#/sqrt(DTE)

        DTE_vect = np.append(DTE_vect, DTE)
        ROR_vect = np.append(ROR_vect, ROR)

    if DTE_vect.size == 0 or ROR_vect.size == 0:
        return None

    # Calculate the index where FOM_vect is the highest
    max_ROR_index = np.argmax(ROR_vect)
    # optimum_ROR = ROR_vect[max_ROR_index]

    # Find the corresponding value in DTE_vect
    optimum_DTE = DTE_vect[max_ROR_index]

    return optimum_DTE

def find_opt_DTE_call(filtered_data, current_price, delta_long, VIX, \
                    DTE_min, DTE_max, spread_length, slippage_abs, slippage_per, \
                        t_df):

    DTE_vect = np.array([])
    ROR_vect = np.array([])
    flag = False

    for DTE_min in range(DTE_min, DTE_max+1, 1):

        DTE = int(DTE_min)
        DTE_prev = int(DTE_min)
        while(1):

            # Obtain the variance required
            var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)

            DTE_call, ROR_call = find_call_possible_profit(\
                filtered_data, var_long, current_price, DTE_min, std_dev, spread_length,\
                    slippage_abs, slippage_per, t_df)

            if DTE_call is None:
                flag = True
                break
            
            DTE = int(DTE_call)

            if DTE == DTE_prev:
                break
            DTE_prev = DTE

        if flag == True:
            break

        ROR = ROR_call#/sqrt(DTE)

        DTE_vect = np.append(DTE_vect, DTE)
        ROR_vect = np.append(ROR_vect, ROR)

    if DTE_vect.size == 0 or ROR_vect.size == 0:
        return None

    # Calculate the index where FOM_vect is the highest
    max_ROR_index = np.argmax(ROR_vect)
    # optimum_ROR = ROR_vect[max_ROR_index]

    # Find the corresponding value in DTE_vect
    optimum_DTE = DTE_vect[max_ROR_index]

    return optimum_DTE

def find_opt_DTE_put(filtered_data, current_price, delta_long, VIX, \
                    DTE_min, DTE_max, spread_length, slippage_abs, slippage_per, \
                        t_df):

    DTE_vect = np.array([])
    ROR_vect = np.array([])
    flag = False

    for DTE_min in range(DTE_min, DTE_max+1, 1):

        DTE = int(DTE_min)
        DTE_prev = int(DTE_min)
        while(1):

            # Obtain the variance required
            var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)

            DTE_put, ROR_put = find_put_possible_profit(\
                filtered_data, var_long, current_price, DTE_min, std_dev, spread_length,\
                    slippage_abs, slippage_per, t_df)

            if DTE_put is None:
                flag = True
                break
            
            DTE = int(DTE_put)

            if DTE == DTE_prev:
                break
            DTE_prev = DTE

        if flag == True:
            break

        ROR = ROR_put#/sqrt(DTE)

        DTE_vect = np.append(DTE_vect, DTE)
        ROR_vect = np.append(ROR_vect, ROR)

    if DTE_vect.size == 0 or ROR_vect.size == 0:
        return None

    # Calculate the index where FOM_vect is the highest
    max_ROR_index = np.argmax(ROR_vect)
    # optimum_ROR = ROR_vect[max_ROR_index]

    # Find the corresponding value in DTE_vect
    optimum_DTE = DTE_vect[max_ROR_index]

    return optimum_DTE

## Garch functions
# Here's our function to compute the sigmas given the initial guess
def compute_squared_sigmas(X, initial_sigma, theta):
    
    a0 = abs(theta[0])
    a1 = abs(theta[1])
    b1 = abs(theta[2])
    
    T = len(X)
    sigma2 = np.ndarray(T)
    
    sigma2[0] = initial_sigma ** 2
    
    for t in range(1, T):
        # Here's where we apply the equation
        sigma2[t] = a0 + a1*X[t-1]**2 + b1*sigma2[t-1]
    
    return sigma2

def negative_log_likelihood(X, theta):
    
    T = len(X)
    
    # Estimate initial sigma squared
    initial_sigma = np.sqrt(np.mean(X ** 2))
    
    # Generate the squared sigma values
    sigma2 = compute_squared_sigmas(X, initial_sigma, theta)
    
    # Now actually compute
    return -sum(
        [-np.log(np.sqrt(2.0 * np.pi)) -
        (X[t] ** 2) / (2.0 * sigma2[t]) -
        0.5 * np.log(sigma2[t]) for
         t in range(T)]
    )
    
def solve_garch_1_1(X):

    X = X[~np.isnan(X)]

    # Make our objective function by plugging X into our log likelihood function
    objective = partial(negative_log_likelihood, X)

    # Define the constraints for our minimizer
    def constraint1(theta):
        return np.array([0.8 - (theta[1] + theta[2])])

    def constraint2(theta):
        return np.array([theta[1]])

    def constraint3(theta):
        return np.array([theta[2]])

    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3})

    # Actually do the minimization
    result = scipy.optimize.minimize(objective, (1, 0.5, 0.5),
                            method='SLSQP',
                            constraints = cons)
    [a0, a1, b1] = np.abs(result.x)
    sigma1 = sqrt(a0/(1-a1-b1))    
    
    return a0, a1, b1, sigma1

def predict_GARCH(returns_df, current_date, VIX_prev):

    window = 22

    try:
        i_end = returns_df.loc[returns_df['Date'] == current_date].index[0]
        X = np.array(returns_df['Returns'].iloc[i_end-window:i_end-1])

        a0, a1, b1, sigma1 = solve_garch_1_1(X)
        X_prev = (VIX_prev/100) * sqrt(1/365)

        if np.isnan(X_prev):
            X_prev = 0

        sigma_new = sqrt(a0 + b1*sigma1**2 + a1*X_prev**2) 
        sigma_max = sqrt(1/365)
        
        sigma_new = min(sigma_max, sigma_new)

        VIX_new = sigma_new * sqrt(365/1) * 100.0

    except IndexError:
        VIX_new = 0
    
    return VIX_new

## Calculation functions
def calculate_variance(delta_long, DTE, VIX, t_df):

    std_dev = (VIX/100)*sqrt(DTE/365)

    # Find corresponding values using the inverse CDF (percent point function)
    var_long = t.ppf(delta_long, t_df, loc=0, scale=std_dev)

    return var_long, std_dev

def calc_expected_alpha(short_strike, long_strike, short_price, long_price, current_price, std_dev, t_df):

    var_short = abs(short_strike-current_price)/current_price
    delta_short = t.cdf(var_short, t_df, loc=0, scale=std_dev)
    ROR_short = (short_price - long_price)/abs(short_strike-long_strike)
    alpha_short = delta_short*ROR_short

    var_long = abs(long_strike-current_price)/current_price
    delta_long = t.cdf(var_long, t_df, loc=0, scale=std_dev)
    ROR_long = -1.0 + ROR_short
    alpha_long = (1-delta_long)*ROR_long

    var_inbetween = np.linspace(var_short,var_long,101)
    delta_inbetween = t.cdf(var_inbetween, t_df, loc=0, scale=std_dev)
    ROR_inbetween = np.linspace(ROR_short,ROR_long,100)
    alpha_inbetween = np.sum((delta_inbetween[1:]-delta_inbetween[:-1])*ROR_inbetween)

    alpha = alpha_short + alpha_long + alpha_inbetween
    
    return alpha

def find_closest_row(data, condition):
    # Calculate the absolute differences based on the condition
    absolute_differences = data.apply(condition, axis=1)

    # Find the index of the row with the minimum absolute difference
    closest_row_index = absolute_differences.idxmin()

    # Get the closest row
    closest_row = data.loc[closest_row_index]

    return closest_row

def calc_number_contracts(capital_available_start, allocation_fraction, short_strike_call,\
                             long_strike_call, short_strike_put, long_Strike_put, contract_multiplier):
    
    cost_per_contract_c = number_contracts_each(\
        short_strike_call, long_strike_call, contract_multiplier)
    cost_per_contract_p = number_contracts_each(\
        short_strike_put, long_Strike_put, contract_multiplier)
    
    if cost_per_contract_c is None or cost_per_contract_p is None:
        return 0.0, 0.0
    else:
        cost_per_contract_max = max(cost_per_contract_c, cost_per_contract_p)
    
    number_contracts = floor((capital_available_start*allocation_fraction)/cost_per_contract_max)
    amount_for_trade = number_contracts*cost_per_contract_max

    return amount_for_trade, number_contracts
    

def number_contracts_each(short_strike, long_strike, contract_multiplier):

    cost_per_share = abs(long_strike - short_strike)

    if cost_per_share == 0:
        return None

    cost_per_contract = cost_per_share*contract_multiplier

    return cost_per_contract

def call_profit_per_contract(price_at_expiration, short_strike, long_strike, short_price, long_price, contract_multiplier):

    short_received = short_price - max(0, price_at_expiration - short_strike)
    long_received = max(0, price_at_expiration - long_strike) - long_price

    total_per_share = short_received + long_received
    profit_per_contract = total_per_share*contract_multiplier
    premium_received_per_contract = (short_price-long_price)*contract_multiplier

    return premium_received_per_contract, profit_per_contract

def put_profit_per_contract(price_at_expiration, short_strike, long_strike, short_price, long_price, contract_multiplier):

    short_received = short_price - max(0, short_strike - price_at_expiration)
    long_received = max(0, long_strike - price_at_expiration) - long_price

    total_per_share = short_received + long_received
    profit_per_contract = total_per_share*contract_multiplier
    premium_received_per_contract = (short_price-long_price)*contract_multiplier

    return premium_received_per_contract, profit_per_contract

def conditions_open(price_at_expiration_c, short_strike_c, long_strike_c, short_price_c, long_price_c,\
                    price_at_expiration_p, short_strike_p, long_strike_p, short_price_p, long_price_p):

    # Call
    short_received_c = short_price_c - max(0, price_at_expiration_c - short_strike_c)
    long_received_c = max(0, price_at_expiration_c - long_strike_c) - long_price_c
    total_per_share_c = short_received_c + long_received_c
    cost_per_share_c = abs(short_strike_c - long_strike_c)

    # Put
    short_received_p = short_price_p - max(0, short_strike_p - price_at_expiration_p)
    long_received_p = max(0, long_strike_p - price_at_expiration_p) - long_price_p
    total_per_share_p = short_received_p + long_received_p
    cost_per_share_p = abs(short_strike_p - long_strike_p)

    # Total
    premium_received_c = (short_price_c-long_price_c)
    premium_received_p = (short_price_p-long_price_p)
    if premium_received_c <= 0 or premium_received_p <= 0:
        return 0, 0, 0, 0
    premium_received = premium_received_c + premium_received_p
    risk = max(cost_per_share_c, cost_per_share_p) - premium_received
    ROR_possible = premium_received/risk * 100
    ROR_end = (total_per_share_c+total_per_share_p)/risk * 100

    return ROR_end, ROR_possible, premium_received, risk

def conditions_open_call(price_at_expiration_c, short_strike_c, long_strike_c, short_price_c, long_price_c):

    # Call
    short_received_c = short_price_c - max(0, price_at_expiration_c - short_strike_c)
    long_received_c = max(0, price_at_expiration_c - long_strike_c) - long_price_c
    total_per_share_c = short_received_c + long_received_c
    cost_per_share_c = abs(short_strike_c - long_strike_c)

    # Total
    premium_received_c = (short_price_c-long_price_c)
    premium_received = premium_received_c
    risk = cost_per_share_c - premium_received
    ROR_possible = premium_received/risk * 100
    ROR_end = total_per_share_c/risk * 100

    return ROR_end, ROR_possible, premium_received, risk

def conditions_open_put(price_at_expiration_p, short_strike_p, long_strike_p, short_price_p, long_price_p):

    # Put
    short_received_p = short_price_p - max(0, short_strike_p - price_at_expiration_p)
    long_received_p = max(0, long_strike_p - price_at_expiration_p) - long_price_p
    total_per_share_p = short_received_p + long_received_p
    cost_per_share_p = abs(short_strike_p - long_strike_p)

    # Total
    premium_received_p = (short_price_p-long_price_p)
    premium_received = premium_received_p
    risk = cost_per_share_p - premium_received
    ROR_possible = premium_received/risk * 100
    ROR_end = total_per_share_p/risk * 100

    return ROR_end, ROR_possible, premium_received, risk

def slippage_decrease(premium, slippage_abs, slippage_per):

    premium_1 = premium - slippage_abs
    premium_2 = premium * (1 - slippage_per/100)

    premium = max(min(premium_1, premium_2),0)

    return premium

def slippage_increase(cost_to_close, slippage_abs, slippage_per):

    cost_to_close_1 = cost_to_close + slippage_abs
    cost_to_close_2 = cost_to_close*(1 + slippage_per/100)

    cost_to_close = max(cost_to_close_1, cost_to_close_2, 0)

    return cost_to_close

def conditions_close(filter1, date, short_strike_c, long_strike_c,\
                     short_strike_p, long_strike_p, current_underlying_price, slippage_abs, slippage_per):

    filter2 = filter1[filter1['[QUOTE_DATE]'] == date]
    if filter2.empty:
        return None, None
    
    DTE = filter2['[DTE]'].values[0]

    short_row_c = filter2[filter2['[STRIKE]']==short_strike_c]
    if short_row_c.empty:
        short_price_c = 0
    else:
        short_price_c = short_row_c['[C_BID]'].values[0] 
    long_row_c = filter2[filter2['[STRIKE]']==long_strike_c]
    if long_row_c.empty:
        long_price_c = 0
    else:
        long_price_c = long_row_c['[C_ASK]'].values[0] 

    short_row_p = filter2[filter2['[STRIKE]']==short_strike_p]
    if short_row_p.empty:
        short_price_p = 0
    else:
        short_price_p = short_row_p['[P_BID]'].values[0] 
    long_row_p = filter2[filter2['[STRIKE]']==long_strike_p]
    if long_row_p.empty:
        long_price_p = 0
    else:
        long_price_p = long_row_p['[P_ASK]'].values[0] 

    # Apply slippage
    short_price_c = slippage_decrease(short_price_c, slippage_abs, slippage_per)
    long_price_c = slippage_increase(long_price_c, slippage_abs, slippage_per)
    short_price_p = slippage_decrease(short_price_p, slippage_abs, slippage_per)
    long_price_p = slippage_increase(long_price_p, slippage_abs, slippage_per)

    # Call
    short_received_c = max(0, current_underlying_price - short_strike_c) - short_price_c
    long_received_c = long_price_c - max(0, current_underlying_price - long_strike_c)
    total_per_share_c = min(0,short_received_c + long_received_c)

    # Put
    short_received_p = max(0, short_strike_p - current_underlying_price) - short_price_p
    long_received_p = long_price_p - max(0, long_strike_p - current_underlying_price)
    total_per_share_p = min(0,short_received_p + long_received_p)

    # Total
    cost_to_close = total_per_share_c + total_per_share_p

    return cost_to_close, DTE

def get_underlying_values_at_date(dataframe, target_date):

    try:        
        # Locate the row with the specified date and extract the 'Price' value
        price = dataframe.loc[dataframe['Date'] == target_date, 'Price'].values[0]

        # Locate the row with the specified date and extract the 'VIX' value
        vix = dataframe.loc[dataframe['Date'] == target_date, 'VIX'].values[0]
        
        return price, vix

    except IndexError as error:
        # Handle the case where the specified date is not in the DataFrame
        return None, None

def find_call_strike_and_prices(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                current_price, spread_length, slippage_abs, slippage_per):
    
    # Find the row where [C_DELTA] is closest to delta-long
    strike_desired_long = current_price*(1 + var_long)
    condition_long = lambda row: abs(row['[STRIKE]'] - strike_desired_long)
    closest_long_row = find_closest_row(min_dte_row, condition_long)
    long_price = closest_long_row['[C_BID]'] 
    long_strike = closest_long_row['[STRIKE]']

    # Ensure that closest_long_row is at least 1 index above closest_short_row    
    try:
        closest_short_row = min_dte_row.iloc[min_dte_row.index.get_loc(closest_long_row.name) - int(spread_length)]
        short_price = closest_short_row['[C_ASK]'] 
        short_strike = closest_short_row['[STRIKE]']
    except IndexError:
        return expire_date_no_trade, None, None, None, None, None
        
    # Check if short_strike and long_strike exist
    if pd.isnull(short_strike) or pd.isnull(long_strike):
        return expire_date_no_trade, None, None, None, None, None  # Short strike or long strike does not exist
    
    # Check if short_price is less than or equal to long_price
    if short_price <= long_price:
        return expire_date_no_trade, None, None, None, None, None  # Short price is less than or equal to long price
    
    # Get the expire_date
    expire_date = min_dte_row['[EXPIRE_DATE]'].values[0]

    # Find the value of [UNDERLYING_LAST] at the specified expire_date
    price_expiration, skip = get_underlying_values_at_date(price_df, expire_date)
    if price_expiration is None:
        return expire_date_no_trade, None, None, None, None, None
    
    # Apply slippage
    short_price = slippage_decrease(short_price, slippage_abs, slippage_per)
    long_price = slippage_increase(long_price, slippage_abs, slippage_per)
    
    return expire_date, short_strike, long_strike, short_price, long_price, price_expiration

def find_put_strike_and_prices(min_dte_row, var_long, price_df,\
                               expire_date_no_trade, current_price, spread_length,\
                                  slippage_abs, slippage_per):
    
    # Find the row where [C_DELTA] is closest to delta-long
    strike_desired_long = current_price*(1 - var_long)
    condition_long = lambda row: abs(row['[STRIKE]'] - strike_desired_long)
    closest_long_row = find_closest_row(min_dte_row, condition_long)
    long_price = closest_long_row['[P_BID]'] 
    long_strike = closest_long_row['[STRIKE]']

    # Ensure that closest_long_row is at least 1 index above closest_short_row    
    try:
        closest_short_row = min_dte_row.iloc[min_dte_row.index.get_loc(closest_long_row.name) + int(spread_length)]
        short_price = closest_short_row['[P_ASK]'] 
        short_strike = closest_short_row['[STRIKE]']
    except IndexError:
        return expire_date_no_trade, None, None, None, None, None    

    # Check if short_strike and long_strike exist
    if pd.isnull(short_strike) or pd.isnull(long_strike):
        return expire_date_no_trade, None, None, None, None, None  # Short strike or long strike does not exist
    
    
    # Check if short_price is less than or equal to long_price
    if short_price <= long_price:
        return expire_date_no_trade, None, None, None, None, None  # Short price is less than or equal to long price
    
    # Get the expire_date
    expire_date = min_dte_row['[EXPIRE_DATE]'].values[0]
    
    # Find the value of [UNDERLYING_LAST] at the specified expire_date
    price_expiration, skip = get_underlying_values_at_date(price_df, expire_date)

    if price_expiration is None:
        return expire_date_no_trade, None, None, None, None, None
    
    # Apply slippage
    short_price = slippage_decrease(short_price, slippage_abs, slippage_per)
    long_price = slippage_increase(long_price, slippage_abs, slippage_per)

    return expire_date, short_strike, long_strike, short_price, long_price, price_expiration

def calc_iron_condor(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction):

    expire_date_call, short_strike_call, long_strike_call, short_price_call,\
        long_price_call, price_at_expiration_call =\
    find_call_strike_and_prices(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per)
    
    expire_date_put, short_strike_put, long_strike_put, short_price_put,\
          long_price_put, price_at_expiration_put =\
        find_put_strike_and_prices(min_dte_row, var_long, price_df,\
                                   expire_date_no_trade, current_price, spread_length,\
                                    slippage_abs, slippage_per)      

    if None in [expire_date_call, short_strike_call, long_strike_call, short_price_call,\
          long_price_call, price_at_expiration_call,expire_date_put, short_strike_put, long_strike_put, short_price_put,\
          long_price_put, price_at_expiration_put]:
        return expire_date_no_trade, 0, np.nan, np.nan
    
    # Latest date calcs
    expire_date = max(expire_date_call, expire_date_put)

    # Expected alpha (EV%)
    alpha_expected_call = calc_expected_alpha(short_strike_call, long_strike_call, short_price_call,\
                                              long_price_call, current_price, std_dev, t_df)
    alpha_expected_put = calc_expected_alpha(short_strike_put, long_strike_put, short_price_put,\
                                             long_price_put, current_price, std_dev, t_df)
    alpha_expected = alpha_expected_call + alpha_expected_put

    if np.isnan(alpha_expected) or alpha_expected >= 1:
        return expire_date_no_trade, 0, np.nan, np.nan

    # What actually happens
    ROR_end, ROR_possible, premium_received, risk = conditions_open(\
        price_at_expiration_call, short_strike_call,\
                                     long_strike_call, short_price_call, long_price_call,\
                    price_at_expiration_put, short_strike_put,\
                        long_strike_put, short_price_put, long_price_put)
       
    if np.isnan(ROR_possible) or ROR_possible >= 100 or ROR_possible <= 0:
        return expire_date_no_trade, 0, np.nan, np.nan

    ROR_percentage = ROR_end
    date = expire_date

    # Calculate required outputs
    contract_multiplier = 100
    amount_for_trade, number_contracts = \
        calc_number_contracts(capital_available_start, allocation_fraction, short_strike_call,\
                            long_strike_call, short_strike_put, long_strike_put, contract_multiplier)
    
    # What actually happened
    trade_profit = (ROR_percentage/100)*amount_for_trade - 0.65*number_contracts

    return date, trade_profit, ROR_percentage, alpha_expected

def calc_call(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction):

    expire_date, short_strike, long_strike, short_price,\
        long_price, price_at_expiration =\
    find_call_strike_and_prices(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per)
 

    if None in [expire_date, short_strike, long_strike, short_price,\
          long_price, price_at_expiration]:
        return expire_date_no_trade, 0, np.nan, np.nan

    # Expected alpha (EV%)
    alpha_expected_call = calc_expected_alpha(short_strike, long_strike, short_price,\
                                              long_price, current_price, std_dev, t_df)
    alpha_expected = alpha_expected_call

    if np.isnan(alpha_expected) or alpha_expected >= 1:
        return expire_date_no_trade, 0, np.nan, np.nan

    # What actually happens
    ROR_end, ROR_possible, premium_received, risk = conditions_open_call(\
        price_at_expiration, short_strike,\
                                     long_strike, short_price, long_price)
       
    if np.isnan(ROR_possible) or ROR_possible >= 100 or ROR_possible <= 0:
        return expire_date_no_trade, 0, np.nan, np.nan

    ROR_percentage = ROR_end
    date = expire_date

    # Calculate required outputs
    contract_multiplier = 100
    cost_per_contract = number_contracts_each(short_strike, long_strike, contract_multiplier)
    number_contracts = floor((capital_available_start*allocation_fraction)/cost_per_contract)
    amount_for_trade = number_contracts*cost_per_contract
    
    # What actually happened
    trade_profit = (ROR_percentage/100)*amount_for_trade - 0.65*number_contracts

    return date, trade_profit, ROR_percentage, alpha_expected

def calc_put(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction):

    expire_date, short_strike, long_strike, short_price,\
        long_price, price_at_expiration =\
    find_put_strike_and_prices(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per) 

    if None in [expire_date, short_strike, long_strike, short_price,\
          long_price, price_at_expiration]:
        return expire_date_no_trade, 0, np.nan, np.nan

    # Expected alpha (EV%)
    alpha_expected = calc_expected_alpha(short_strike, long_strike, short_price,\
                                              long_price, current_price, std_dev, t_df)

    if np.isnan(alpha_expected) or alpha_expected >= 1:
        return expire_date_no_trade, 0, np.nan, np.nan

    # What actually happens
    ROR_end, ROR_possible, premium_received, risk = conditions_open_put(\
        price_at_expiration, short_strike,\
                                     long_strike, short_price, long_price)
       
    if np.isnan(ROR_possible) or ROR_possible >= 100 or ROR_possible <= 0:
        return expire_date_no_trade, 0, np.nan, np.nan

    ROR_percentage = ROR_end
    date = expire_date

    # Calculate required outputs
    contract_multiplier = 100
    cost_per_contract = number_contracts_each(short_strike, long_strike, contract_multiplier)
    number_contracts = floor((capital_available_start*allocation_fraction)/cost_per_contract)
    amount_for_trade = number_contracts*cost_per_contract
    
    # What actually happened
    trade_profit = (ROR_percentage/100)*amount_for_trade - 0.65*number_contracts

    return date, trade_profit, ROR_percentage, alpha_expected

def calc_trade_profit(data, price_df, returns_df, capital_available_start, current_date,\
                      allocation_fraction, alpha_required, slippage_abs,\
                         slippage_per, t_df, VIX_prev, optimization_paramaters):
    
    # Special case of only DTE variation
    DTE_min, DTE_max, spread_length, max_loss_probability = optimization_paramaters
    
    # High VIX
    expire_date_no_trade = current_date + np.timedelta64(1, 'D')
    
    # Filter rows for the specified start_date and DTE_min
    filtered_data = data[data['[QUOTE_DATE]'] == current_date]
    current_price, _ = get_underlying_values_at_date(price_df, current_date)
    VIX = predict_GARCH(returns_df, current_date, VIX_prev)

    if filtered_data.empty or current_price is None or VIX==0 or np.isnan(VIX):
        return expire_date_no_trade, capital_available_start, np.nan, np.nan, np.nan


    # Iron Condor
    delta_long = 1.0 - max_loss_probability/2

    # Find the row with the minimum [DTE] value in filtered_data
    optimum_DTE = find_opt_DTE(filtered_data, current_price,\
                               delta_long, VIX, DTE_min, DTE_max,\
                                spread_length, slippage_abs, slippage_per, t_df)
    filtered_data = filtered_data[filtered_data['[DTE]'] >= optimum_DTE]
    min_dte_row = filtered_data[filtered_data['[DTE]'] == filtered_data['[DTE]'].min()]
    
    if min_dte_row.empty:
        return expire_date_no_trade, capital_available_start, np.nan, np.nan, np.nan
    
    # Get the DTE value used
    DTE = min_dte_row['[DTE]'].values[0]

    # Actual calculation
    var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)
    result_iron_condor = \
        calc_iron_condor(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction)
    
    # Call
    delta_long = 1.0 - max_loss_probability

    # Find the row with the minimum [DTE] value in filtered_data
    optimum_DTE = find_opt_DTE_call(filtered_data, current_price,\
                               delta_long, VIX, DTE_min, DTE_max,\
                                spread_length, slippage_abs, slippage_per, t_df)
    filtered_data = filtered_data[filtered_data['[DTE]'] >= optimum_DTE]
    min_dte_row = filtered_data[filtered_data['[DTE]'] == filtered_data['[DTE]'].min()]
    
    if min_dte_row.empty:
        return expire_date_no_trade, capital_available_start, np.nan, np.nan, np.nan
    
    # Get the DTE value used
    DTE = min_dte_row['[DTE]'].values[0]

    # Actual calculation
    var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)
    result_call = \
        calc_call(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction)
    
    # Put
    delta_long = 1.0 - max_loss_probability

    # Find the row with the minimum [DTE] value in filtered_data
    optimum_DTE = find_opt_DTE_put(filtered_data, current_price,\
                               delta_long, VIX, DTE_min, DTE_max,\
                                spread_length, slippage_abs, slippage_per, t_df)
    filtered_data = filtered_data[filtered_data['[DTE]'] >= optimum_DTE]
    min_dte_row = filtered_data[filtered_data['[DTE]'] == filtered_data['[DTE]'].min()]
    
    if min_dte_row.empty:
        return expire_date_no_trade, capital_available_start, np.nan, np.nan, np.nan
    
    # Get the DTE value used
    DTE = min_dte_row['[DTE]'].values[0]

    var_long, std_dev = calculate_variance(delta_long, DTE, VIX, t_df)
    result_put = \
        calc_put(min_dte_row, var_long, price_df, expire_date_no_trade,\
                                    current_price, spread_length, slippage_abs, slippage_per,\
                                        std_dev, t_df, alpha_required,\
                                            capital_available_start, allocation_fraction)
    
    ROR_vect = np.array([result_iron_condor[3], result_call[3], result_put[3]])
    max_index = np.argmax(ROR_vect)
    
    if max_index == 0:
        date = result_iron_condor[0]
        trade_profit = result_iron_condor[1]
        ROR_percentage = result_iron_condor[2]
        alpha_expected = result_iron_condor[3]

    elif max_index == 1:
        date = result_call[0]
        trade_profit = result_call[1]
        ROR_percentage = result_call[2]
        alpha_expected = result_call[3]

    elif max_index == 2:
        date = result_put[0]
        trade_profit = result_put[1]
        ROR_percentage = result_put[2]
        alpha_expected = result_put[3]

    if alpha_expected < alpha_required:
        return expire_date_no_trade, capital_available_start, np.nan, np.nan, np.nan

    capital_available_new = capital_available_start + trade_profit

    return date, capital_available_new, ROR_percentage, alpha_expected*100.0, VIX

## Loop function
def backtest_func(options_df, price_df, returns_df, start_date, alpha_required,\
                  allocation_fraction, capital_available_start,\
                        slippage_abs, slippage_per, t_df, optimization_paramaters):
    
    # Find the last date:
    # Step 1: Find the most recent [QUOTE_DATE]
    latest_date = options_df['[QUOTE_DATE]'].max()

    VIX_prev = 0

    current_date = start_date
    capital_available = capital_available_start
    current_date_vect = [current_date]
    capital_available_vect = [capital_available]
    ROR_actual_vect = [0]
    ROR_expect_vect = [0]

    while(1):
        
        current_date, capital_available, ROR_actual, ROR_expect, VIX_prev = calc_trade_profit(\
        options_df, price_df, returns_df, capital_available, current_date, allocation_fraction,\
                        alpha_required, slippage_abs, slippage_per, t_df, VIX_prev, optimization_paramaters)

        current_date_vect = np.append(current_date_vect, current_date)
        capital_available_vect = np.append(capital_available_vect, capital_available)
        ROR_actual_vect = np.append(ROR_actual_vect, ROR_actual)
        ROR_expect_vect = np.append(ROR_expect_vect, ROR_expect)

        if capital_available <= 0 or current_date >= latest_date:
            break


    return current_date_vect, capital_available_vect, ROR_actual_vect, ROR_expect_vect