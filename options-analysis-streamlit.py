import sys
import subprocess

# Install packages if not already installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt

# Import the backend function
from functions import backend

def main():
    st.title('Options Analysis Dashboard')

    # Sidebar for Inputs
    st.sidebar.header('Analysis Parameters')
    
    # Input widgets with default values
    DTE_max = st.sidebar.number_input('Maximum Days to Expiration (DTE)', 
                                min_value=0, 
                                value=1)
    DTE_min = st.sidebar.number_input('Minimum Days to Expiration (DTE)', 
                                min_value=0,
                                value=0)
    volume_min = st.sidebar.number_input('Minimum Volume', 
                                         min_value=0, 
                                         value=0)
    required_probability_per = st.sidebar.number_input('Required Probability Percentage', 
                                                       min_value=0.0, 
                                                       max_value=100.0, 
                                                       value=0.0)
    required_EV_per = st.sidebar.number_input('Required Expected Value Percentage', 
                                              min_value=-100.0, 
                                              max_value=100.0, 
                                              value=0.0)

    # Load data
    tickers = ["SPY", "QQQ"]
    tickers_underlying = ["^GSPC", "^NDX"]

    # Read pickle files
    options_df_SPY = pd.read_pickle(tickers[0]+"_options_data.pkl")
    options_df_QQQ = pd.read_pickle(tickers[1]+"_options_data.pkl")

    # Run backend function when user clicks a button
    if st.sidebar.button('Run Analysis'):
        # Process data
        df_buy_call_mean_array, df_buy_put_mean_array = backend(
            DTE_max, DTE_min, volume_min, 
            required_probability_per, required_EV_per,
            options_df_SPY, options_df_QQQ, 
            tickers, tickers_underlying
        )

        # Create figure with matplotlib
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 16})
        plt.suptitle('SPY - Average Return of Trades - Predicted versus Actual Values')

        # Calls - Normal Distribution
        plt.subplot(2,2,1)
        plt.plot(df_buy_call_mean_array[0,0].index, df_buy_call_mean_array[0,0]['EV% mean'], label='Predicted return')
        plt.plot(df_buy_call_mean_array[0,0].index, df_buy_call_mean_array[0,0]['actual return % mean'], label='Actual return')
        plt.legend()
        plt.grid()
        plt.xlabel('Date Latest Trade is Opened [Year]')
        plt.ylabel('Average Return [%]')
        plt.title('Normal Distribution - Calls')

        # Calls - Student's t Distribution
        plt.subplot(2,2,2)
        plt.plot(df_buy_call_mean_array[0,1].index, df_buy_call_mean_array[0,1]['EV% mean'], label='Predicted return')
        plt.plot(df_buy_call_mean_array[0,1].index, df_buy_call_mean_array[0,1]['actual return % mean'], label='Actual return')
        plt.legend()
        plt.grid()
        plt.xlabel('Date Latest Trade is Opened [Year]')
        plt.ylabel('Average Return [%]')
        plt.title("Student's t Distribution - Calls")

        # Puts - Normal Distribution
        plt.subplot(2,2,3)
        plt.plot(df_buy_put_mean_array[0,0].index, df_buy_put_mean_array[0,0]['EV% mean'], label='Predicted return')
        plt.plot(df_buy_put_mean_array[0,0].index, df_buy_put_mean_array[0,0]['actual return % mean'], label='Actual return')
        plt.legend()
        plt.grid()
        plt.xlabel('Date Latest Trade is Opened [Year]')
        plt.ylabel('Average Return [%]')
        plt.title('Normal Distribution - Puts')

        # Puts - Student's t Distribution
        plt.subplot(2,2,4)
        plt.plot(df_buy_put_mean_array[0,1].index, df_buy_put_mean_array[0,1]['EV% mean'], label='Predicted return')
        plt.plot(df_buy_put_mean_array[0,1].index, df_buy_put_mean_array[0,1]['actual return % mean'], label='Actual return')
        plt.legend()
        plt.grid()
        plt.xlabel('Date Latest Trade is Opened [Year]')
        plt.ylabel('Average Return [%]')
        plt.title("Student's t Distribution - Puts")

        plt.tight_layout()
        
        # Display plot in Streamlit
        st.pyplot(plt)

        # Optional: Display some statistics or additional information
        st.subheader('Analysis Summary')
        st.write(f"Parameters Used:")
        st.write(f"- Max DTE: {DTE_max}")
        st.write(f"- Min DTE: {DTE_min}")
        st.write(f"- Min Volume: {volume_min}")
        st.write(f"- Required Probability %: {required_probability_per}")
        st.write(f"- Required Expected Value %: {required_EV_per}")

if __name__ == "__main__":
    main()
