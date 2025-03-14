from src.live_market_data import LiveMarketData

live_data = LiveMarketData()
live_df = live_data.fetch_live_data()
print("Live Data:\n", live_df)
