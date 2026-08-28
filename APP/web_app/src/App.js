import React, { useEffect, useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import './App.css';
import CryptoChart from './components/chart';
import CoinDropdown from './components/dropDown';
import NewsCard from './components/newsCard';
import {
  fetchBackendHealthThunk,
  fetchCoinMetadataThunk,
  fetchNews,
  selectHealth,
  selectHealthError,
  selectHealthLoading,
  selectMetadata,
  selectMetadataError,
  selectMetadataLoading,
  selectNews,
  selectNewsError,
  selectNewsLoading,
  selectPrediction,
  selectPredictionError,
  selectPredictionLoading,
} from './redux/store';

const classAccent = {
  Buy: 'bg-green-500',
  Neutral: 'bg-amber-400',
  Sell: 'bg-red-500',
};

const PredictionCard = ({ prediction, predictionError, predictionLoading, health, healthLoading, healthError }) => {
  const topProbability = Number.isFinite(prediction?.confidence_percentage)
    ? Math.max(0, Math.min(100, prediction.confidence_percentage))
    : 0;
  const signal = predictionError ? 'Unavailable' : prediction?.prediction_label || 'Unknown';
  const accentClass = classAccent[signal] || 'bg-gray-500';

  if (predictionLoading) {
    return <p className="text-center text-gray-500">Loading model prediction...</p>;
  }

  return (
    <div className="rounded-lg border bg-white p-4 shadow-sm">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <div className="text-sm font-medium text-gray-600">Model Signal</div>
          <div className="text-2xl font-semibold text-gray-900">
            {signal}
          </div>
        </div>
        <span className="text-2xl font-semibold text-gray-900">{Math.round(topProbability)}%</span>
      </div>

      <div className="h-3 overflow-hidden rounded-full bg-gray-200">
        <div
          className={`h-full transition-all ${accentClass}`}
          style={{ width: `${topProbability}%` }}
        />
      </div>

      {prediction?.model_name && (
        <div className="mt-4 text-xs text-gray-500">
          {prediction.model_name} · {prediction.lookback_days}-day window
        </div>
      )}

      {prediction?.input_summary?.target_trade_day && (
        <div className="mt-1 text-xs text-gray-500">
          Predicting {prediction.input_summary.target_trade_day} from {prediction.input_summary.latest_feature_day}
        </div>
      )}

      {predictionError && <div className="mt-3 text-sm text-red-500">{predictionError}</div>}

      {prediction?.warnings?.length > 0 && (
        <div className="mt-3 rounded bg-amber-50 p-3 text-sm text-amber-700">
          {prediction.warnings.join(' ')}
        </div>
      )}

      <div className="mt-4 border-t pt-3 text-xs text-gray-500">
        {healthLoading && <div>Checking backend health...</div>}
        {healthError && <div className="text-red-500">{healthError}</div>}
        {!healthLoading && !healthError && health && (
          <div>
            Backend: {health.model_loaded ? 'ready' : 'degraded'}
            {health.model_name ? ` · ${health.model_name}` : ''}
          </div>
        )}
      </div>
    </div>
  );
};

const App = () => {
  const [selectedCoin, setSelectedCoin] = useState('BTCUSDT');
  const [selectedRange, setSelectedRange] = useState('7d');
  const metadata = useSelector(selectMetadata);
  const metadataLoading = useSelector(selectMetadataLoading);
  const metadataError = useSelector(selectMetadataError);
  const newsData = useSelector(selectNews);
  const newsLoading = useSelector(selectNewsLoading);
  const newsError = useSelector(selectNewsError);
  const prediction = useSelector(selectPrediction);
  const predictionLoading = useSelector(selectPredictionLoading);
  const predictionError = useSelector(selectPredictionError);
  const health = useSelector(selectHealth);
  const healthLoading = useSelector(selectHealthLoading);
  const healthError = useSelector(selectHealthError);
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(fetchBackendHealthThunk());
  }, [dispatch]);

  useEffect(() => {
    dispatch(fetchNews(selectedCoin));
    dispatch(fetchCoinMetadataThunk(selectedCoin));
  }, [dispatch, selectedCoin]);

  const calculateDateRange = (range) => {
    const now = Date.now();
    switch (range) {
      case '5d':
        return { startDate: now - 5 * 24 * 60 * 60 * 1000, endDate: now };
      case '1w':
        return { startDate: now - 7 * 24 * 60 * 60 * 1000, endDate: now };
      case '2w':
        return { startDate: now - 14 * 24 * 60 * 60 * 1000, endDate: now };
      case '1m':
        return { startDate: now - 30 * 24 * 60 * 60 * 1000, endDate: now };
      default:
        return { startDate: now - 7 * 24 * 60 * 60 * 1000, endDate: now };
    }
  };

  const { startDate, endDate } = useMemo(() => calculateDateRange(selectedRange), [selectedRange]);
  const chartKey = useMemo(() => `${selectedCoin}-${startDate}-${endDate}`, [selectedCoin, startDate, endDate]);

  return (
    <div className="flex min-h-screen bg-gray-900">
      <div className="w-2/5 bg-gray-200 p-4">
        {metadataLoading ? (
          <p className="text-center text-gray-500">Loading metadata...</p>
        ) : metadataError ? (
          <p className="text-center text-red-500">Error: {metadataError}</p>
        ) : metadata ? (
          <div className="grid grid-cols-1 gap-4">
            <div className="rounded-lg bg-white p-4 shadow">
              <h3 className="text-sm font-semibold text-gray-700">Market Cap</h3>
              <p className="text-lg font-bold text-gray-900">${Number(metadata.marketCap).toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-white p-4 shadow">
              <h3 className="text-sm font-semibold text-gray-700">Price Change (24h)</h3>
              <p className={`text-lg font-bold ${metadata.priceChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {metadata.priceChange}%
              </p>
            </div>
            <div className="rounded-lg bg-white p-4 shadow">
              <h3 className="text-sm font-semibold text-gray-700">24h High</h3>
              <p className="text-lg font-bold text-gray-900">${Number(metadata.highPrice).toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-white p-4 shadow">
              <h3 className="text-sm font-semibold text-gray-700">24h Low</h3>
              <p className="text-lg font-bold text-gray-900">${Number(metadata.lowPrice).toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-white p-4 shadow">
              <h3 className="text-sm font-semibold text-gray-700">Volume (24h)</h3>
              <p className="text-lg font-bold text-gray-900">${Number(metadata.volume).toLocaleString()}</p>
            </div>
          </div>
        ) : (
          <p className="text-center text-gray-500">No metadata available</p>
        )}
      </div>

      <div className="mx-auto max-w-3xl flex-1 p-5">
        <div className="mb-5 flex flex-wrap justify-between">
          <div className="mr-5 flex flex-col">
            <CoinDropdown coins={['BTCUSDT', 'ETHUSDT']} onChange={setSelectedCoin} />
          </div>
          <div className="flex flex-col">
            <select
              value={selectedRange}
              onChange={(e) => setSelectedRange(e.target.value)}
              className="rounded bg-transparent py-1 text-white"
            >
              <option value="5d">5 Days</option>
              <option value="1w">1 Week</option>
              <option value="2w">2 Weeks</option>
              <option value="1m">1 Month</option>
            </select>
          </div>
        </div>
        <div className="min-h-[500px] w-full overflow-x-auto rounded-lg border bg-white p-4 shadow-sm">
          <div className="w-[1000px]">
            <CryptoChart
              key={chartKey}
              symbol={selectedCoin}
              endDate={endDate}
              startDate={startDate}
            />
          </div>
        </div>
      </div>

      <div className="w-2/5 bg-gray-200 p-4">
        <h2 className="mb-4 text-lg font-semibold">Community Sentiment</h2>
        <div className="mb-4">
          <PredictionCard
            prediction={prediction}
            predictionError={predictionError}
            predictionLoading={predictionLoading}
            health={health}
            healthLoading={healthLoading}
            healthError={healthError}
          />
        </div>
        <h3 className="mb-2 text-md font-semibold">Crypto News</h3>
        <div className="h-64 min-h-96 overflow-y-auto border-t">
          {newsLoading ? (
            <p className="mt-4 text-center text-gray-500">Loading news...</p>
          ) : newsError ? (
            <p className="mt-4 text-center text-red-500">Error fetching news: {newsError}</p>
          ) : newsData.length === 0 ? (
            <p className="mt-4 text-center text-gray-500">No news available.</p>
          ) : (
            newsData.map((item, index) => (
              <NewsCard
                key={index}
                title={item.title}
                description={item.description}
                thumbnail={item.thumbnail}
                url={item.url}
                createdAt={item.createdAt}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
