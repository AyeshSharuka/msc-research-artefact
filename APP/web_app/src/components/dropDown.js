// src/components/CoinDropdown.js
import React from 'react';

const CoinDropdown = ({ coins, onChange }) => {
    return (
        <select onChange={(e) => onChange(e.target.value)} className="mb-4 bg-transparent text-white">
            {coins.map((coin) => (
                <option key={coin} value={coin}>
                    {coin}
                </option>
            ))}
        </select>
    );
};

export default CoinDropdown;
