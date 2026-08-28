// src/components/DateDropdown.js
import React, { useState } from 'react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

const DateDropdown = ({ onChange }) => {
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date());

    const handleDateChange = (dates) => {
        const [start, end] = dates;
        setStartDate(start);
        setEndDate(end);

        // Call the onChange handler with the selected date range
        if (start && end) {
            onChange({ startDate: start.getTime(), endDate: end.getTime() });
        }
    };

    return (
        <div className="mb-4">
            <DatePicker
                selected={startDate}
                onChange={handleDateChange}
                startDate={startDate}
                endDate={endDate}
                selectsRange
                inline
            />
        </div>
    );
};

export default DateDropdown;
