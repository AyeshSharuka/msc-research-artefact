import React, { useEffect, useRef } from 'react';
import { Chart as Chartjs } from 'chart.js/auto';
import _ from 'lodash';
import { useDispatch, useSelector } from 'react-redux';

import { fetchData, selectChartData, selectChartError, selectChartLoading } from '../redux/store';

const Chart = ({ symbol, startDate, endDate }) => {
  const dispatch = useDispatch();
  const data = useSelector(selectChartData);
  const loading = useSelector(selectChartLoading);
  const error = useSelector(selectChartError);
  const chartContainer = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    dispatch(fetchData({ symbol, startDate, endDate }));
  }, [dispatch, symbol, startDate, endDate]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    if (data && chartContainer.current) {
      chartInstance.current = new Chartjs(chartContainer.current, {
        type: 'line',
        data: _.cloneDeep(data),
        options: {
          responsive: true,
          plugins: {
            legend: { position: 'top' },
            title: {
              display: true,
              text: 'Closing Prices Over Time',
            },
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Date',
              },
            },
            y: {
              title: {
                display: true,
                text: 'Closing Price',
              },
            },
          },
        },
      });
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [data, symbol, startDate, endDate]);

  if (loading) {
    return <p>Loading chart...</p>;
  }

  if (error) {
    return <p>{error}</p>;
  }

  return <canvas ref={chartContainer} />;
};

export default Chart;
