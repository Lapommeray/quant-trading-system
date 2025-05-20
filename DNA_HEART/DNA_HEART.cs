
using QuantConnect.Algorithm;
using QuantConnect.Data.Market;
using QuantConnect.Indicators;
using System.Collections.Generic;

namespace QMP.Modules
{
    public class DNAHeart
    {
        private readonly RollingWindow<decimal> sentimentWindow;
        private readonly RollingWindow<decimal> priceWindow;
        private decimal emotionalSignal;

        public DNAHeart()
        {
            sentimentWindow = new RollingWindow<decimal>(10);
            priceWindow = new RollingWindow<decimal>(10);
            emotionalSignal = 0;
        }

        public void Update(Slice data, decimal socialSentiment)
        {
            if (!data.Bars.ContainsKey("SPY")) return;

            var price = data.Bars["SPY"].Close;

            priceWindow.Add(price);
            sentimentWindow.Add(socialSentiment);

            if (sentimentWindow.IsReady && priceWindow.IsReady)
            {
                var recentSentiment = sentimentWindow[0];
                var recentPrice = priceWindow[0];
                var prevSentiment = sentimentWindow[1];
                var prevPrice = priceWindow[1];

                // Detect emotional divergence
                if (recentSentiment > 0.5m && recentPrice < prevPrice)
                    emotionalSignal = -1; // market feels bullish but price is dropping (trap)
                else if (recentSentiment < -0.5m && recentPrice > prevPrice)
                    emotionalSignal = 1;  // market is fearful but price is rising (fade fear)
                else
                    emotionalSignal = 0;
            }
        }

        public int GetSignal()
        {
            return emotionalSignal > 0 ? 1 : emotionalSignal < 0 ? -1 : 0;
        }
    }
}
