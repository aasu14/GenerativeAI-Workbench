from accurate_analysis import *

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 accurate_analysis.py <STOCK_NAME>")
        print("Example: python3 accurate_analysis.py ADANIGREEN")
        print("Example: python3 accurate_analysis.py RELIANCE")
        return
    
    stock_name = sys.argv[1].upper()
    
    print(f"🚀 Analyzing {stock_name}...")
    print("📊 Fetching real market data from Google Finance & Yahoo Finance...")
    
    # Get accurate data with real current price
    data, current_price, data_source = get_accurate_stock_data(stock_name)
    
    # Perform comprehensive analysis
    technical = comprehensive_technical_analysis(data)
    news = realistic_news_sentiment(stock_name)
    market = realistic_market_analysis()
    recommendation = generate_smart_recommendation(technical, news, market, stock_name)
    price_targets = calculate_smart_price_targets(technical['current_price'], technical, recommendation, stock_name)
    risk = assess_risk(technical, market, recommendation)
    
    # Display results
    display_comprehensive_analysis(stock_name, technical, news, market, recommendation, price_targets, risk, data_source)

if __name__ == '__main__':
    main()
