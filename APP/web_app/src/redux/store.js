import { configureStore, createAsyncThunk, createSlice } from '@reduxjs/toolkit';

import {
  fetchBackendHealth,
  fetchCoinMetadata,
  fetchCryptoData,
  fetchNewsPrediction,
} from '../api/api';

const API_KEY = 'your_api_key';
const BASE_URL = 'https://api.coinfeeds.io/coins';

const toAssetName = (selectedCoin) => (selectedCoin === 'BTCUSDT' ? 'bitcoin' : 'ethereum');

export const fetchTweets = createAsyncThunk(
  'data/fetchTweets',
  async (coinName, { rejectWithValue }) => {
    try {
      const response = await fetch(`${BASE_URL}/${coinName}/tweets?symbol=true`, {
        headers: { 'x-api-key': API_KEY },
      });
      const data = await response.json();
      return data;
    } catch (error) {
      return rejectWithValue('Error fetching tweets');
    }
  },
);

export const fetchNews = createAsyncThunk(
  'data/fetchAndAnalyzeNews',
  async (selectedCoin, { dispatch, rejectWithValue }) => {
    try {
      const response = await fetch('https://cryptocurrency-news2.p.rapidapi.com/v1/cryptodaily', {
        method: 'GET',
        headers: {
          'x-rapidapi-host': 'cryptocurrency-news2.p.rapidapi.com',
          'x-rapidapi-key': '519b19d366msh3c0962a21631e7dp1622cdjsn9fccdab48dd5',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch news');
      }

      const data = await response.json();
      const keyword = selectedCoin === 'BTCUSDT' ? 'bitcoin' : 'ethereum';
      const asset = toAssetName(selectedCoin);

      const filteredNews = data.data
        .filter((news) => {
          const title = (news.title || '').toLowerCase();
          const description = (news.description || '').toLowerCase();
          return title.includes(keyword) || description.includes(keyword);
        })
        .map((news) => ({
          text: [news.title, news.description].filter(Boolean).join('. '),
          date: new Date(news.createdAt).toISOString(),
          source: 'news',
        }));

      dispatch(analyzeFilteredNews({ asset, news: filteredNews }));
      return data.data;
    } catch (error) {
      return rejectWithValue('Error fetching and analyzing news');
    }
  },
);

export const fetchCoinMetadataThunk = createAsyncThunk(
  'data/fetchCoinMetadata',
  async (symbol, { rejectWithValue }) => {
    try {
      return await fetchCoinMetadata(symbol);
    } catch (error) {
      return rejectWithValue('Error fetching coin metadata');
    }
  },
);

export const fetchData = createAsyncThunk(
  'data/fetchData',
  async ({ symbol, startDate, endDate }, { rejectWithValue }) => {
    try {
      const response = await fetchCryptoData(symbol, startDate, endDate);
      return {
        labels: response.map((item) => new Date(item[0]).toLocaleDateString()),
        datasets: [
          {
            label: 'Closing Price',
            data: response.map((item) => parseFloat(item[4])),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
          },
        ],
      };
    } catch (error) {
      return rejectWithValue('Error fetching data');
    }
  },
);

export const analyzeFilteredNews = createAsyncThunk(
  'data/analyzeFilteredNews',
  async (payload, { rejectWithValue }) => {
    try {
      return await fetchNewsPrediction(payload);
    } catch (error) {
      return rejectWithValue(error.message || 'Error analyzing filtered news');
    }
  },
);

export const fetchBackendHealthThunk = createAsyncThunk(
  'data/fetchBackendHealth',
  async (_, { rejectWithValue }) => {
    try {
      return await fetchBackendHealth();
    } catch (error) {
      return rejectWithValue('Error checking backend health');
    }
  },
);

const initialState = {
  prices: null,
  tweets: [],
  news: [],
  metadata: null,
  health: null,
  prediction: {
    prediction_label: 'Unknown',
    predicted_label_numeric: 1,
    probabilities: { Buy: 0, Neutral: 0, Sell: 0 },
    confidence_percentage: 0,
    model_name: null,
    lookback_days: 0,
    input_summary: {},
    warnings: [],
  },
  priceLoading: false,
  newsLoading: false,
  metadataLoading: false,
  predictionLoading: false,
  healthLoading: false,
  priceError: null,
  newsError: null,
  metadataError: null,
  predictionError: null,
  healthError: null,
};

const dataSlice = createSlice({
  name: 'data',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchData.pending, (state) => {
        state.priceLoading = true;
        state.priceError = null;
      })
      .addCase(fetchData.fulfilled, (state, action) => {
        state.prices = action.payload;
        state.priceLoading = false;
      })
      .addCase(fetchData.rejected, (state, action) => {
        state.priceError = action.payload;
        state.priceLoading = false;
      })
      .addCase(fetchNews.pending, (state) => {
        state.newsLoading = true;
        state.newsError = null;
      })
      .addCase(fetchNews.fulfilled, (state, action) => {
        state.news = action.payload;
        state.newsLoading = false;
      })
      .addCase(fetchNews.rejected, (state, action) => {
        state.newsError = action.payload;
        state.newsLoading = false;
      })
      .addCase(fetchCoinMetadataThunk.pending, (state) => {
        state.metadataLoading = true;
        state.metadataError = null;
      })
      .addCase(fetchCoinMetadataThunk.fulfilled, (state, action) => {
        state.metadata = action.payload;
        state.metadataLoading = false;
      })
      .addCase(fetchCoinMetadataThunk.rejected, (state, action) => {
        state.metadataError = action.payload;
        state.metadataLoading = false;
      })
      .addCase(analyzeFilteredNews.pending, (state) => {
        state.predictionLoading = true;
        state.predictionError = null;
      })
      .addCase(analyzeFilteredNews.fulfilled, (state, action) => {
        state.prediction = action.payload;
        state.predictionLoading = false;
      })
      .addCase(analyzeFilteredNews.rejected, (state, action) => {
        state.predictionError = action.payload;
        state.predictionLoading = false;
      })
      .addCase(fetchBackendHealthThunk.pending, (state) => {
        state.healthLoading = true;
        state.healthError = null;
      })
      .addCase(fetchBackendHealthThunk.fulfilled, (state, action) => {
        state.health = action.payload;
        state.healthLoading = false;
      })
      .addCase(fetchBackendHealthThunk.rejected, (state, action) => {
        state.healthError = action.payload;
        state.healthLoading = false;
      })
      .addCase(fetchTweets.fulfilled, (state, action) => {
        state.tweets = action.payload;
      })
      .addCase(fetchTweets.rejected, (state, action) => {
        state.newsError = action.payload;
      });
  },
});

export const store = configureStore({
  reducer: dataSlice.reducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      thunk: true,
      serializableCheck: false,
    }),
});

export const selectChartData = (state) => state.prices;
export const selectChartLoading = (state) => state.priceLoading;
export const selectChartError = (state) => state.priceError;
export const selectNews = (state) => state.news;
export const selectNewsLoading = (state) => state.newsLoading;
export const selectNewsError = (state) => state.newsError;
export const selectMetadata = (state) => state.metadata;
export const selectMetadataLoading = (state) => state.metadataLoading;
export const selectMetadataError = (state) => state.metadataError;
export const selectPrediction = (state) => state.prediction;
export const selectPredictionLoading = (state) => state.predictionLoading;
export const selectPredictionError = (state) => state.predictionError;
export const selectHealth = (state) => state.health;
export const selectHealthLoading = (state) => state.healthLoading;
export const selectHealthError = (state) => state.healthError;

export default store;
