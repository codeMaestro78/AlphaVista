# This is a temporary fix for the market cap storage issue
# The following line needs to be replaced in the original file:

# OLD:
# 'Market Cap (B)': '₹' + str(round(market_cap_bn, 2)) if symbol.endswith('.NS') else '$' + str(round(market_cap_bn, 2)),

# NEW:
# 'Market Cap (B)': market_cap_bn,  # Numeric value for filtering
# 'Market Cap': market_cap,         # Formatted string with currency symbol
