import streamlit as st
import mibian
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Options Gamma Exposure Calculator",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Options Gamma Exposure (GEX) Calculator")
st.markdown("""
    This app calculates and visualizes the Gamma Exposure (GEX) profile for a given ticker symbol.
    GEX helps identify potential price levels where market maker hedging activity might influence price action.
""")

# Sidebar for inputs
st.sidebar.header("Inputs")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", "SPY")
contract_multiplier = st.sidebar.number_input("Contract Multiplier", min_value=1, value=100)
gex_threshold = st.sidebar.number_input("GEX Threshold (for flip detection)", min_value=1e-8, value=1e-6, format="%.8f")
custom_expiry = st.sidebar.checkbox("Select specific expiration date?", False)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_spot_price(ticker_symbol):
    """
    Fetch the current spot price for the given ticker symbol using Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        spot_price = ticker.history(period="1d")['Close'].iloc[-1]
        return spot_price
    except Exception as e:
        st.error(f"Error fetching spot price: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_available_expiration_dates(ticker_symbol):
    """
    Get available option expiration dates for the ticker.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        options = ticker.options
        return options
    except Exception as e:
        st.error(f"Error fetching expiration dates: {e}")
        return []

def calculate_gamma(spot_price, strike_price, implied_volatility, days_to_expiration):
    """
    Calculate the Gamma of an option using the Black-Scholes formula.
    """
    try:
        # Make sure we have at least 1 day to expiration for calculations
        days_to_expiration = max(1, days_to_expiration)
        
        # Check for valid inputs to prevent errors
        if (strike_price <= 0 or implied_volatility <= 0 or spot_price <= 0):
            return 0
            
        # mibian expects volatility as a percentage (e.g., 20 for 20%)
        implied_volatility_percent = implied_volatility * 100
        
        option = mibian.BS([spot_price, strike_price, 0, days_to_expiration], 
                           volatility=implied_volatility_percent)
        
        # Get gamma (this is per dollar change in the underlying)
        gamma = option.gamma / 100  # Convert to decimal form
        
        return gamma
    except Exception as e:
        st.warning(f"Error calculating gamma: {e} for S={spot_price}, K={strike_price}, σ={implied_volatility}, t={days_to_expiration}")
        return 0

def calculate_gex(gamma, open_interest, spot_price, contract_multiplier=100):
    """
    Calculate Gamma Exposure (GEX) in dollar terms.
    
    GEX = Gamma * OI * Contract Multiplier * Spot^2 * 0.01
    
    The 0.01 represents a 1% move in the underlying.
    """
    if open_interest <= 0 or np.isnan(open_interest):
        return 0
    gex = gamma * open_interest * contract_multiplier * spot_price * spot_price * 0.01
    return gex

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_option_data(ticker_symbol, selected_expiration=None):
    """
    Fetch option data for the given ticker symbol and optionally a specific expiration date.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        options = ticker.options
        
        if len(options) == 0:
            st.error(f"No options data available for {ticker_symbol}")
            return None, None, None
        
        # If a specific expiration is selected, use it
        if selected_expiration and selected_expiration in options:
            expiration_date = selected_expiration
        else:
            # Get the nearest expiration date that's not today
            for expiration_date in options:
                days_to_expiration = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days
                if days_to_expiration > 0:
                    break
            else:
                # If all expirations are today or in the past, use the next one
                expiration_date = options[0]
        
        # Get calls and puts for the expiration date
        option_chain = ticker.option_chain(expiration_date)
        calls = option_chain.calls
        puts = option_chain.puts
        
        return calls, puts, expiration_date
    except Exception as e:
        st.error(f"Error fetching option data: {e}")
        return None, None, None

def find_gamma_flip(strike_gex_data, threshold=1e-6):
    """
    Find the gamma flip point based on GEX values.
    Returns the strike price at which the cumulative GEX changes from positive to negative.
    """
    # Sort by strike price
    sorted_data = sorted(strike_gex_data, key=lambda x: x[0])
    
    # Compute cumulative GEX at each strike
    strikes = [x[0] for x in sorted_data]
    gex_values = [x[1] for x in sorted_data]
    
    # Only consider meaningful GEX values
    filtered_data = [(strike, gex) for strike, gex in zip(strikes, gex_values) 
                      if abs(gex) > threshold]
    
    if not filtered_data:
        return None
    
    # Check for sign changes in cumulative GEX
    cum_gex = 0
    prev_sign = None
    
    for strike, gex in filtered_data:
        cum_gex += gex
        
        # Skip very small values
        if abs(cum_gex) < threshold:
            continue
            
        current_sign = 1 if cum_gex > 0 else -1
        
        if prev_sign is not None and current_sign != prev_sign:
            return strike
            
        prev_sign = current_sign
    
    return None

def plot_gex_profile_plotly(ticker_symbol, gex_data, flip_point, spot_price):
    """
    Create a Plotly figure for the GEX profile.
    """
    if not gex_data:
        return None
    
    # Sort by strike
    sorted_data = sorted(gex_data, key=lambda x: x[0])
    strikes = [x[0] for x in sorted_data]
    gex_values = [x[1] for x in sorted_data]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Strike': strikes,
        'GEX': gex_values
    })
    
    # Compute cumulative GEX
    df['Cumulative GEX'] = df['GEX'].cumsum()
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Gamma Exposure by Strike", "Cumulative Gamma Exposure"))
    
    # Add GEX by strike - color bars based on positive/negative
    colors = ['green' if g > 0 else 'red' for g in df['GEX']]
    
    fig.add_trace(
        go.Bar(
            x=df['Strike'],
            y=df['GEX'],
            marker_color=colors,
            name='GEX'
        ),
        row=1, col=1
    )
    
    # Add cumulative GEX
    fig.add_trace(
        go.Scatter(
            x=df['Strike'],
            y=df['Cumulative GEX'],
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Cumulative GEX'
        ),
        row=2, col=1
    )
    
    # Add zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # Add current spot price
    fig.add_vline(x=spot_price, line_dash="dash", line_color="black", 
                  annotation_text=f"Spot: {spot_price:.2f}", 
                  annotation_position="top right")
    
    # Add gamma flip point if found
    if flip_point:
        fig.add_vline(x=flip_point, line_dash="dash", line_color="red", 
                     annotation_text=f"Flip: {flip_point:.2f}", 
                     annotation_position="bottom right")
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Gamma Exposure (GEX) Profile for {ticker_symbol}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2=dict(title="Strike Price"),
        yaxis=dict(title="GEX"),
        yaxis2=dict(title="Cumulative GEX")
    )
    
    return fig

def create_gex_table(gex_data):
    """
    Create a table with GEX data.
    """
    if not gex_data:
        return None
        
    # Sort by absolute GEX value
    sorted_data = sorted(gex_data, key=lambda x: abs(x[1]), reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(sorted_data, columns=['Strike', 'GEX'])
    
    # Format values
    df['Strike'] = df['Strike'].map(lambda x: f"{x:.2f}")
    df['GEX'] = df['GEX'].map(lambda x: f"{x:,.2f}")
    
    return df

def main():
    # Get available expiration dates
    expiration_dates = get_available_expiration_dates(ticker_symbol)
    
    selected_expiration = None
    if custom_expiry and expiration_dates:
        selected_expiration = st.sidebar.selectbox(
            "Select Expiration Date", 
            expiration_dates,
            index=0
        )
    
    # Calculate button
    if st.sidebar.button("Calculate GEX"):
        # Show a spinner while loading data
        with st.spinner("Fetching market data..."):
            spot_price = get_spot_price(ticker_symbol)
            
            if spot_price is None:
                st.error(f"Could not fetch data for ticker: {ticker_symbol}")
                return
                
            calls, puts, expiration_date = get_option_data(ticker_symbol, selected_expiration)
            
            if calls is None or puts is None:
                st.error("Failed to fetch options data")
                return
                
            days_to_expiration = max(1, (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days)
        
        st.header(f"GEX Analysis for {ticker_symbol}")
        st.subheader(f"Expiration: {expiration_date} ({days_to_expiration} days)")
        
        # Display the spot price
        st.metric("Current Spot Price", f"${spot_price:.2f}")
        
        # Metric containers for summary stats
        col1, col2, col3 = st.columns(3)
        
        # Process option data
        gex_data = []
        total_call_gex = 0
        total_put_gex = 0
        call_gex_count = 0
        put_gex_count = 0
        
        # Progress bar for calculation
        progress_bar = st.progress(0)
        total_options = len(calls) + len(puts)
        progress_step = 1 / total_options if total_options > 0 else 1
        progress_value = 0
        
        # Calculate Gamma Exposure for Calls
        for call in calls.itertuples():
            strike_price = call.strike
            implied_volatility = call.impliedVolatility
            
            # Skip options with missing or invalid data
            if (np.isnan(strike_price) or np.isnan(implied_volatility) or 
                strike_price <= 0 or implied_volatility <= 0):
                progress_value += progress_step
                progress_bar.progress(progress_value)
                continue
                
            open_interest = call.openInterest if not np.isnan(call.openInterest) else 0
            
            if open_interest <= 0:
                progress_value += progress_step
                progress_bar.progress(progress_value)
                continue  # Skip options with no open interest
            
            gamma = calculate_gamma(spot_price, strike_price, implied_volatility, days_to_expiration)
            gex = calculate_gex(gamma, open_interest, spot_price, contract_multiplier)
            
            gex_data.append((strike_price, gex))  # Calls have positive gamma
            total_call_gex += gex
            call_gex_count += 1
            
            progress_value += progress_step
            progress_bar.progress(progress_value)
        
        # Calculate Gamma Exposure for Puts
        for put in puts.itertuples():
            strike_price = put.strike
            implied_volatility = put.impliedVolatility
            
            # Skip options with missing or invalid data
            if (np.isnan(strike_price) or np.isnan(implied_volatility) or 
                strike_price <= 0 or implied_volatility <= 0):
                progress_value += progress_step
                progress_bar.progress(progress_value)
                continue
                
            open_interest = put.openInterest if not np.isnan(put.openInterest) else 0
            
            if open_interest <= 0:
                progress_value += progress_step
                progress_bar.progress(progress_value)
                continue  # Skip options with no open interest
            
            gamma = calculate_gamma(spot_price, strike_price, implied_volatility, days_to_expiration)
            # Note: Puts have negative gamma impact on market
            gex = calculate_gex(gamma, open_interest, spot_price, contract_multiplier) * -1
            
            gex_data.append((strike_price, gex))
            total_put_gex += gex
            put_gex_count += 1
            
            progress_value += progress_step
            progress_bar.progress(progress_value)
        
        # Complete progress bar
        progress_bar.progress(1.0)
        
        # Find Gamma Flip point
        flip_point = find_gamma_flip(gex_data, threshold=gex_threshold)
        
        # Display summary metrics
        with col1:
            st.metric("Total Call GEX", f"${total_call_gex:,.2f}")
        with col2:
            st.metric("Total Put GEX", f"${total_put_gex:,.2f}")
        with col3:
            net_gex = total_call_gex + total_put_gex
            st.metric("Net GEX", f"${net_gex:,.2f}", 
                     delta="Positive" if net_gex > 0 else "Negative",
                     delta_color="normal")
        
        # Display Gamma Flip point
        if flip_point:
            st.success(f"Gamma Flip Point: ${flip_point:.2f}")
        else:
            st.warning("No Gamma Flip Point detected with current threshold")
        
        # Plot GEX profile
        if gex_data:
            st.subheader("GEX Profile Visualization")
            fig = plot_gex_profile_plotly(ticker_symbol, gex_data, flip_point, spot_price)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display top GEX values
            st.subheader("Largest GEX Values by Strike")
            df = create_gex_table(gex_data)
            if df is not None:
                st.dataframe(df.head(15), use_container_width=True)
        else:
            st.error("No meaningful GEX data found. This could be due to low open interest or missing data.")
        
    else:
        # Display instructions when the app first loads
        st.info("Enter a ticker symbol and click 'Calculate GEX' to analyze options gamma exposure.")
        st.markdown("""
        ### What is Gamma Exposure (GEX)?
        
        GEX measures the expected change in market makers' delta hedging requirements for a 1% move in the underlying asset.
        
        ### What GEX Values Tell Us:
        
        1. **Market Sensitivity** - High GEX values indicate areas where small price movements could trigger significant hedging activity
        
        2. **Price Support/Resistance** - Positive GEX creates stabilizing force (market makers sell into rallies, buy dips)
        
        3. **Gamma Flip Point** - Price level where cumulative GEX changes from positive to negative
        
        4. **Options Positioning** - Concentration of GEX at specific strikes reveals where large positions exist
        
        5. **Volatility Expectations** - Large GEX suggests potentially lower realized volatility in that price range
        """)

if __name__ == "__main__":
    main()
