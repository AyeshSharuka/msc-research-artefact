import React from 'react';

const FALLBACK_THUMBNAIL =
  'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96"><rect width="100%" height="100%" fill="%23e5e7eb"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="%236b7280" font-size="12">News</text></svg>';

const NewsCard = ({ title, description, thumbnail, url, createdAt }) => {
  return (
    <div className="flex border-b p-4 last:border-none">
      <img
        src={thumbnail || FALLBACK_THUMBNAIL}
        alt={title || 'News'}
        className="mr-4 h-16 w-16 rounded object-cover"
      />
      <div className="flex-1">
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold text-blue-500"
        >
          {title || 'Untitled article'}
        </a>
        <p className="mt-1 text-sm text-gray-600">{description || 'No description available.'}</p>
        <div className="mt-2 text-xs text-gray-500">
          {createdAt ? new Date(createdAt).toLocaleString() : 'Unknown publish time'}
        </div>
      </div>
    </div>
  );
};

export default NewsCard;
