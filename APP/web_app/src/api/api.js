import axios from 'axios';

const BASE_URL = 'https://api.binance.com/api/v3';
const BACKEND_URL = 'http://localhost:8000';

export const fetchCryptoData = async (symbol, startDate, endDate) => {
  try {
    const response = await axios.get(`${BASE_URL}/klines`, {
      params: {
        symbol,
        interval: '1d',
        startTime: startDate,
        endTime: endDate,
      },
    });
    return response.data;
  } catch (error) {
    throw new Error('Error fetching data');
  }
};

export const fetchCoinMetadata = async (symbol) => {
  try {
    const response = await axios.get(`${BASE_URL}/ticker/24hr`, {
      params: { symbol },
    });
    return {
      marketCap: response.data.quoteVolume,
      priceChange: response.data.priceChangePercent,
      highPrice: response.data.highPrice,
      lowPrice: response.data.lowPrice,
      volume: response.data.volume,
    };
  } catch (error) {
    throw new Error('Error fetching coin metadata');
  }
};

export const fetchBackendHealth = async () => {
  try {
    const response = await axios.get(`${BACKEND_URL}/health`);
    return response.data;
  } catch (error) {
    throw new Error('Error checking backend health');
  }
};

export const fetchNewsPrediction = async ({ asset, news }) => {
  try {
    const response = await axios.post(`${BACKEND_URL}/predict`, {
      asset,
      news,
    });
    return response.data;
  } catch (error) {
    const message = error.response?.data?.detail || 'Error sending news data to backend';
    throw new Error(message);
  }
};
