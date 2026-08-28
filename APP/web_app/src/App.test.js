import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';

jest.mock('./components/chart', () => () => <div>Chart Placeholder</div>);
jest.mock('./api/api', () => ({
  fetchBackendHealth: jest.fn().mockResolvedValue({
    model_loaded: true,
    model_name: 'hierarchical_market_sentiment_rnn',
  }),
  fetchCoinMetadata: jest.fn().mockResolvedValue({
    marketCap: 1000000,
    priceChange: 1.25,
    highPrice: 100,
    lowPrice: 90,
    volume: 50000,
  }),
  fetchCryptoData: jest.fn().mockResolvedValue([]),
  fetchNewsPrediction: jest.fn().mockResolvedValue({
    prediction_label: 'Neutral',
    predicted_label_numeric: 1,
    probabilities: { Buy: 0.2, Neutral: 0.5, Sell: 0.3 },
    confidence_percentage: 50,
    model_name: 'hierarchical_market_sentiment_rnn',
    lookback_days: 14,
    input_summary: {},
    warnings: [],
  }),
}));

import App from './App';
import { store } from './redux/store';

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ data: [] }),
  });
});

afterEach(() => {
  jest.resetAllMocks();
});

test('renders the updated dashboard shell', async () => {
  render(
    <Provider store={store}>
      <App />
    </Provider>,
  );

  expect(await screen.findByText(/Community Sentiment/i)).toBeInTheDocument();
  expect(screen.getByText(/Crypto News/i)).toBeInTheDocument();
  expect(screen.getByText(/Chart Placeholder/i)).toBeInTheDocument();
});
