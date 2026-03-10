const { addonBuilder } = require("stremio-addon-sdk");
const fetch = require("node-fetch").default;
const logger = require("./utils/logger");
const path = require("path");
const { decryptConfig } = require("./utils/crypto");
const { withRetry } = require("./utils/apiRetry");
const TMDB_API_BASE = "https://api.themoviedb.org/3";
const TMDB_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
const TMDB_DISCOVER_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
const AI_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
const RPDB_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
const DEFAULT_RPDB_KEY = process.env.RPDB_API_KEY;
const DEFAULT_FANART_KEY = process.env.FANART_API_KEY;
const ENABLE_LOGGING = process.env.ENABLE_LOGGING === "true" || false;
const TRAKT_API_BASE = "https://api.trakt.tv";
const TRAKT_CACHE_DURATION = 24 * 60 * 60 * 1000;
const TRAKT_RAW_DATA_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
const DEFAULT_TRAKT_CLIENT_ID = process.env.TRAKT_CLIENT_ID;
const MAX_AI_RECOMMENDATIONS = 30;

// Default local LLM settings (can be overridden per-user in config)
const DEFAULT_LLM_BASE_URL = process.env.LLM_BASE_URL || "http://localhost:11434/v1";
const DEFAULT_LLM_MODEL = process.env.LLM_MODEL || "llama3.2";
const DEFAULT_LLM_API_KEY = process.env.LLM_API_KEY || "ollama";

let queryCounter = 0;

/**
 * OpenAI-compatible LLM call — works with Ollama, LM Studio, vLLM, etc.
 * @param {string} prompt - The prompt text
 * @param {string} baseUrl - Base URL of the OpenAI-compatible endpoint
 * @param {string} model - Model name to use
 * @param {string} apiKey - API key (can be any string for Ollama)
 * @returns {Promise<string>} - The model's response text
 */
async function callLocalLLM(prompt, baseUrl, model, apiKey) {
  const url = `${baseUrl.replace(/\/$/, "")}/chat/completions`;
  const body = {
    model: model,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.7,
    stream: false,
  };

  const response = await withRetry(
    async () => {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = new Error(`LLM API error: ${res.status}`);
        err.status = res.status;
        throw err;
      }
      return res.json();
    },
    {
      maxRetries: 3,
      initialDelay: 2000,
      maxDelay: 10000,
      shouldRetry: (error) => !error.status || error.status >= 500,
      operationName: "Local LLM API call",
    }
  );

  const text = response?.choices?.[0]?.message?.content?.trim();
  if (!text) throw new Error("Empty response from local LLM");
  return text;
}

class SimpleLRUCache {
  constructor(options = {}) {
    this.max = options.max || 1000;
    this.ttl = options.ttl || Infinity;
    this.cache = new Map();
    this.timestamps = new Map();
    this.expirations = new Map();
  }

  set(key, value) {
    if (this.cache.size >= this.max) {
      const oldestKey = this.timestamps.keys().next().value;
      this.delete(oldestKey);
    }
    this.cache.set(key, value);
    this.timestamps.set(key, Date.now());
    if (this.ttl !== Infinity) {
      const expiration = Date.now() + this.ttl;
      this.expirations.set(key, expiration);
    }
    return this;
  }

  get(key) {
    if (!this.cache.has(key)) return undefined;
    const expiration = this.expirations.get(key);
    if (expiration && Date.now() > expiration) {
      this.delete(key);
      return undefined;
    }
    this.timestamps.delete(key);
    this.timestamps.set(key, Date.now());
    return this.cache.get(key);
  }

  has(key) {
    if (!this.cache.has(key)) return false;
    const expiration = this.expirations.get(key);
    if (expiration && Date.now() > expiration) {
      this.delete(key);
      return false;
    }
    return true;
  }

  delete(key) {
    this.cache.delete(key);
    this.timestamps.delete(key);
    this.expirations.delete(key);
    return true;
  }

  clear() {
    this.cache.clear();
    this.timestamps.clear();
    this.expirations.clear();
    return true;
  }

  get size() { return this.cache.size; }
  keys() { return Array.from(this.cache.keys()); }

  serialize() {
    const entries = [];
    for (const [key, value] of this.cache.entries()) {
      const timestamp = this.timestamps.get(key);
      const expiration = this.expirations.get(key);
      entries.push({ key, value, timestamp, expiration });
    }
    return { max: this.max, ttl: this.ttl, entries };
  }

  deserialize(data) {
    if (!data || !data.entries) return false;
    this.max = data.max || this.max;
    this.ttl = data.ttl || this.ttl;
    this.clear();
    for (const entry of data.entries) {
      if (entry.expiration && Date.now() > entry.expiration) continue;
      this.cache.set(entry.key, entry.value);
      this.timestamps.set(entry.key, entry.timestamp);
      if (entry.expiration) this.expirations.set(entry.key, entry.expiration);
    }
    return true;
  }
}

const tmdbCache = new SimpleLRUCache({ max: 25000, ttl: TMDB_CACHE_DURATION });
const tmdbDetailsCache = new SimpleLRUCache({ max: 25000, ttl: TMDB_CACHE_DURATION });
const aiRecommendationsCache = new SimpleLRUCache({ max: 25000, ttl: AI_CACHE_DURATION });
const rpdbCache = new SimpleLRUCache({ max: 25000, ttl: RPDB_CACHE_DURATION });
const fanartCache = new SimpleLRUCache({ max: 5000, ttl: RPDB_CACHE_DURATION });
const similarContentCache = new SimpleLRUCache({ max: 5000, ttl: AI_CACHE_DURATION });
const traktRawDataCache = new SimpleLRUCache({ max: 1000, ttl: TRAKT_RAW_DATA_CACHE_DURATION });
const traktCache = new SimpleLRUCache({ max: 1000, ttl: TRAKT_CACHE_DURATION });
const tmdbDiscoverCache = new SimpleLRUCache({ max: 1000, ttl: TMDB_DISCOVER_CACHE_DURATION });
const queryAnalysisCache = new SimpleLRUCache({ max: 1000, ttl: AI_CACHE_DURATION });

const HOST = process.env.HOST ? `https://${process.env.HOST}` : "https://stremio.itcon.au";
const PORT = 7000;
const BASE_PATH = "/aisearch";

setInterval(() => {
  logger.info("Cache statistics", {
    tmdbCache: { size: tmdbCache.size, maxSize: tmdbCache.max },
    tmdbDetailsCache: { size: tmdbDetailsCache.size, maxSize: tmdbDetailsCache.max },
    tmdbDiscoverCache: { size: tmdbDiscoverCache.size, maxSize: tmdbDiscoverCache.max },
    aiCache: { size: aiRecommendationsCache.size, maxSize: aiRecommendationsCache.max },
    rpdbCache: { size: rpdbCache.size, maxSize: rpdbCache.max },
  });
}, 60 * 60 * 1000);

function purgeEmptyAiCacheEntries() {
  const cacheKeys = aiRecommendationsCache.keys();
  let purgedCount = 0;
  const totalScanned = cacheKeys.length;
  for (const key of cacheKeys) {
    const cachedItem = aiRecommendationsCache.get(key);
    const recommendations = cachedItem?.data?.recommendations;
    const hasMovies = recommendations?.movies?.length > 0;
    const hasSeries = recommendations?.series?.length > 0;
    if (!hasMovies && !hasSeries) {
      aiRecommendationsCache.delete(key);
      purgedCount++;
    }
  }
  return { scanned: totalScanned, purged: purgedCount, remaining: aiRecommendationsCache.size };
}

function mergeAndDeduplicate(newItems, existingItems) {
  const existingMap = new Map();
  existingItems.forEach((item) => {
    const media = item.movie || item.show;
    const id = item.id || media?.ids?.trakt;
    if (id) existingMap.set(id, item);
  });
  newItems.forEach((item) => {
    const media = item.movie || item.show;
    const id = item.id || media?.ids?.trakt;
    if (id) {
      if (!existingMap.has(id) || (item.last_activity && existingMap.get(id).last_activity && new Date(item.last_activity) > new Date(existingMap.get(id).last_activity))) {
        existingMap.set(id, item);
      }
    }
  });
  return Array.from(existingMap.values());
}

function processGenres(watchedItems, ratedItems) {
  const genres = new Map();
  watchedItems?.forEach((item) => { const media = item.movie || item.show; media.genres?.forEach((genre) => { genres.set(genre, (genres.get(genre) || 0) + 1); }); });
  ratedItems?.forEach((item) => { const media = item.movie || item.show; const weight = item.rating / 5; media.genres?.forEach((genre) => { genres.set(genre, (genres.get(genre) || 0) + weight); }); });
  return Array.from(genres.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([genre, count]) => ({ genre, count }));
}

function processActors(watchedItems, ratedItems) {
  const actors = new Map();
  watchedItems?.forEach((item) => { const media = item.movie || item.show; media.cast?.forEach((actor) => { actors.set(actor.name, (actors.get(actor.name) || 0) + 1); }); });
  ratedItems?.forEach((item) => { const media = item.movie || item.show; const weight = item.rating / 5; media.cast?.forEach((actor) => { actors.set(actor.name, (actors.get(actor.name) || 0) + weight); }); });
  return Array.from(actors.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([actor, count]) => ({ actor, count }));
}

function processDirectors(watchedItems, ratedItems) {
  const directors = new Map();
  watchedItems?.forEach((item) => { const media = item.movie || item.show; media.crew?.forEach((person) => { if (person.job === "Director") directors.set(person.name, (directors.get(person.name) || 0) + 1); }); });
  ratedItems?.forEach((item) => { const media = item.movie || item.show; const weight = item.rating / 5; media.crew?.forEach((person) => { if (person.job === "Director") directors.set(person.name, (directors.get(person.name) || 0) + weight); }); });
  return Array.from(directors.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([director, count]) => ({ director, count }));
}

function processYears(watchedItems, ratedItems) {
  const years = new Map();
  watchedItems?.forEach((item) => { const media = item.movie || item.show; const year = parseInt(media.year); if (year) years.set(year, (years.get(year) || 0) + 1); });
  ratedItems?.forEach((item) => { const media = item.movie || item.show; const year = parseInt(media.year); const weight = item.rating / 5; if (year) years.set(year, (years.get(year) || 0) + weight); });
  if (years.size === 0) return null;
  return { start: Math.min(...years.keys()), end: Math.max(...years.keys()), preferred: Array.from(years.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] };
}

function processRatings(ratedItems) {
  const ratings = new Map();
  ratedItems?.forEach((item) => { ratings.set(item.rating, (ratings.get(item.rating) || 0) + 1); });
  return Array.from(ratings.entries()).sort((a, b) => b[1] - a[1]).map(([rating, count]) => ({ rating, count }));
}

async function processPreferencesInParallel(watched, rated, history) {
  const [genres, actors, directors, yearRange, ratings] = await Promise.all([
    Promise.resolve(processGenres(watched, rated)),
    Promise.resolve(processActors(watched, rated)),
    Promise.resolve(processDirectors(watched, rated)),
    Promise.resolve(processYears(watched, rated)),
    Promise.resolve(processRatings(rated)),
  ]);
  return { genres, actors, directors, yearRange, ratings };
}

function createErrorMeta(title, message) {
  const words = message.split(' ');
  let lines = [];
  let currentLine = words[0] || '';
  for (let i = 1; i < words.length; i++) {
    let testLine = currentLine + ' ' + words[i];
    if (testLine.length > 35) { lines.push(currentLine); currentLine = words[i]; }
    else currentLine = testLine;
  }
  lines.push(currentLine);
  const messageTspans = lines.map((line, index) => `<tspan x="250" y="${560 + index * 30}">${line}</tspan>`).join('');
  const svg = `<svg width="500" height="750" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#2d2d2d" /><path d="M250 50 L450 400 L50 400 Z" fill="#c0392b"/><path d="M250 120 L400 380 L100 380 Z" fill="#e74c3c"/><text fill="white" font-size="60" font-family="Arial, sans-serif" x="250" y="270" text-anchor="middle" font-weight="bold">!</text><text fill="white" font-size="32" font-family="Arial, sans-serif" x="250" y="500" text-anchor="middle" font-weight="bold">${title}</text><text fill="white" font-size="24" font-family="Arial, sans-serif" text-anchor="middle">${messageTspans}</text><text fill="#bdc3c7" font-size="20" font-family="Arial, sans-serif" x="250" y="700" text-anchor="middle">Please check the addon configuration.</text></svg>`;
  const posterDataUri = `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
  return { id: `error:${title.replace(/\s+/g, '_')}`, type: 'movie', name: title, description: message, poster: posterDataUri, posterShape: 'regular' };
}

async function fetchTraktIncrementalData(clientId, accessToken, type, lastUpdate) {
  const startDate = new Date(lastUpdate).toISOString().split(".")[0] + "Z";
  const endpoints = [
    `${TRAKT_API_BASE}/users/me/watched/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
    `${TRAKT_API_BASE}/users/me/ratings/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
    `${TRAKT_API_BASE}/users/me/history/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
  ];
  const headers = { "Content-Type": "application/json", "trakt-api-version": "2", "trakt-api-key": clientId, Authorization: `Bearer ${accessToken}` };
  const responses = await Promise.all(endpoints.map((endpoint) => makeApiCall(endpoint, headers).then((res) => res.json()).catch((err) => { logger.error("Trakt API Error:", { endpoint, error: err.message }); return []; })));
  return { watched: responses[0] || [], rated: responses[1] || [], history: responses[2] || [] };
}

async function fetchTraktWatchedAndRated(clientId, accessToken, type = "movies", config = null) {
  logger.info("fetchTraktWatchedAndRated called", { hasClientId: !!clientId, hasAccessToken: !!accessToken, type });
  if (!clientId || !accessToken) { logger.error("Missing Trakt credentials"); return null; }

  const makeApiCall = async (url, headers) => {
    return await withRetry(async () => {
      const response = await fetch(url, { headers });
      if (response.status === 401) logger.warn("Trakt access token expired.");
      return response;
    }, { maxRetries: 3, baseDelay: 1000, shouldRetry: (error) => !error.status || (error.status !== 401 && error.status !== 403), operationName: "Trakt API call" });
  };

  const rawCacheKey = `trakt_raw_${accessToken}_${type}`;
  const processedCacheKey = `trakt_${accessToken}_${type}`;

  if (traktCache.has(processedCacheKey)) {
    const cached = traktCache.get(processedCacheKey);
    logger.info("Trakt processed cache hit", { cacheKey: processedCacheKey, type });
    return cached.data;
  }

  let rawData;
  let isIncremental = false;

  if (traktRawDataCache.has(rawCacheKey)) {
    const cachedRaw = traktRawDataCache.get(rawCacheKey);
    const lastUpdate = cachedRaw.lastUpdate || cachedRaw.timestamp;
    try {
      const newData = await fetchTraktIncrementalData(clientId, accessToken, type, lastUpdate);
      rawData = {
        watched: mergeAndDeduplicate(newData.watched, cachedRaw.data.watched),
        rated: mergeAndDeduplicate(newData.rated, cachedRaw.data.rated),
        history: mergeAndDeduplicate(newData.history, cachedRaw.data.history),
        lastUpdate: Date.now(),
      };
      isIncremental = true;
      traktRawDataCache.set(rawCacheKey, { timestamp: Date.now(), lastUpdate: Date.now(), data: rawData });
    } catch (error) {
      logger.error("Incremental Trakt update failed, falling back to full refresh", { error: error.message });
      isIncremental = false;
    }
  }

  if (!rawData) {
    try {
      const endpoints = [
        `${TRAKT_API_BASE}/users/me/watched/${type}?extended=full&page=1&limit=100`,
        `${TRAKT_API_BASE}/users/me/ratings/${type}?extended=full&page=1&limit=100`,
        `${TRAKT_API_BASE}/users/me/history/${type}?extended=full&page=1&limit=100`,
      ];
      const headers = { "Content-Type": "application/json", "trakt-api-version": "2", "trakt-api-key": clientId, Authorization: `Bearer ${accessToken}` };
      const responses = await Promise.all(endpoints.map((endpoint) => makeApiCall(endpoint, headers).then((res) => res.json()).catch((err) => { logger.error("Trakt API Error:", { endpoint, error: err.message }); return []; })));
      const [watched, rated, history] = responses;
      rawData = { watched: watched || [], rated: rated || [], history: history || [], lastUpdate: Date.now() };
      traktRawDataCache.set(rawCacheKey, { timestamp: Date.now(), lastUpdate: Date.now(), data: rawData });
    } catch (error) {
      logger.error("Trakt API Error:", { error: error.message, stack: error.stack });
      return null;
    }
  }

  const preferences = await processPreferencesInParallel(rawData.watched, rawData.rated, rawData.history);
  const result = { watched: rawData.watched, rated: rawData.rated, history: rawData.history, preferences, lastUpdate: rawData.lastUpdate, isIncrementalUpdate: isIncremental };
  traktCache.set(processedCacheKey, { timestamp: Date.now(), data: result });
  return result;
}

async function searchTMDB(title, type, year, tmdbKey, language = "en-US", includeAdult = false) {
  const startTime = Date.now();
  const cacheKey = `${title}-${type}-${year}-${language}-adult:${includeAdult}`;
  if (tmdbCache.has(cacheKey)) return tmdbCache.get(cacheKey).data;

  try {
    const searchType = type === "movie" ? "movie" : "tv";
    const searchParams = new URLSearchParams({ api_key: tmdbKey, query: title, year: year, include_adult: includeAdult, language: language });
    const searchUrl = `${TMDB_API_BASE}/search/${searchType}?${searchParams.toString()}`;
    const responseData = await withRetry(async () => {
      const searchResponse = await fetch(searchUrl);
      if (!searchResponse.ok) {
        const errorData = await searchResponse.json().catch(() => ({}));
        let errorMessage = searchResponse.status === 401 ? "Invalid TMDB API key" : searchResponse.status === 429 ? "TMDB API rate limit exceeded" : `TMDB API error: ${searchResponse.status} ${errorData?.status_message || ""}`;
        const error = new Error(errorMessage);
        error.status = searchResponse.status;
        error.isRateLimit = searchResponse.status === 429;
        error.isInvalidKey = searchResponse.status === 401;
        throw error;
      }
      return searchResponse.json();
    }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB search API call", shouldRetry: (error) => !error.isInvalidKey && (!error.status || error.status >= 500 || error.isRateLimit) });

    if (responseData?.results?.[0]) {
      const result = responseData.results[0];
      const tmdbData = { poster: result.poster_path ? `https://image.tmdb.org/t/p/w500${result.poster_path}` : null, backdrop: result.backdrop_path ? `https://image.tmdb.org/t/p/original${result.backdrop_path}` : null, tmdbRating: result.vote_average, genres: result.genre_ids, overview: result.overview || "", tmdb_id: result.id, title: result.title || result.name, release_date: result.release_date || result.first_air_date };

      if (!tmdbData.imdb_id) {
        const detailsCacheKey = `details_${searchType}_${result.id}_${language}`;
        let detailsData;
        if (tmdbDetailsCache.has(detailsCacheKey)) {
          detailsData = tmdbDetailsCache.get(detailsCacheKey).data;
        } else {
          const detailsUrl = `${TMDB_API_BASE}/${searchType}/${result.id}?api_key=${tmdbKey}&append_to_response=external_ids&language=${language}`;
          detailsData = await withRetry(async () => {
            const detailsResponse = await fetch(detailsUrl);
            if (!detailsResponse.ok) throw new Error(`TMDB details API error: ${detailsResponse.status}`);
            return detailsResponse.json();
          }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB details API call" });
          tmdbDetailsCache.set(detailsCacheKey, { timestamp: Date.now(), data: detailsData });
        }
        if (detailsData) tmdbData.imdb_id = detailsData.imdb_id || detailsData.external_ids?.imdb_id;
      }

      tmdbCache.set(cacheKey, { timestamp: Date.now(), data: tmdbData });
      return tmdbData;
    }
    tmdbCache.set(cacheKey, { timestamp: Date.now(), data: null });
    return null;
  } catch (error) {
    logger.error("TMDB Search Error:", { error: error.message, params: { title, type, year } });
    return null;
  }
}

async function searchTMDBExactMatch(title, type, tmdbKey, language = "en-US", includeAdult = false) {
  const cacheKey = `tmdb_search_${title}-${type}-${language}-adult:${includeAdult}`;
  if (tmdbCache.has(cacheKey)) {
    const responseData = tmdbCache.get(cacheKey).data;
    if (responseData && responseData.length > 0) {
      const normalizedTitle = title.toLowerCase().trim();
      const exactMatch = responseData.find((result) => (result.title || result.name || "").toLowerCase().trim() === normalizedTitle);
      return { isExactMatch: !!exactMatch, results: responseData };
    }
    return { isExactMatch: false, results: [] };
  }

  try {
    const searchType = type === "movie" ? "movie" : "tv";
    const searchParams = new URLSearchParams({ api_key: tmdbKey, query: title, include_adult: includeAdult, language: language });
    const searchUrl = `${TMDB_API_BASE}/search/${searchType}?${searchParams.toString()}`;
    const responseData = await withRetry(async () => {
      const searchResponse = await fetch(searchUrl);
      if (!searchResponse.ok) throw new Error(`TMDB search error: ${searchResponse.status}`);
      return searchResponse.json();
    }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB search API call", shouldRetry: (error) => !error.status || error.status >= 500 });

    const results = responseData?.results || [];
    tmdbCache.set(cacheKey, { timestamp: Date.now(), data: results });
    if (results.length > 0) {
      const normalizedTitle = title.toLowerCase().trim();
      const exactMatch = results.find((result) => (result.title || result.name || "").toLowerCase().trim() === normalizedTitle);
      return { isExactMatch: !!exactMatch, results };
    }
    return { isExactMatch: false, results: [] };
  } catch (error) {
    logger.error("TMDB Search Error:", { error: error.message, params: { title, type } });
    return { isExactMatch: false, results: [] };
  }
}

const manifest = {
  id: "au.itcon.aisearch",
  version: "1.0.65-local-llm",
  name: "AI Search (Local LLM)",
  description: "AI-powered movie and series recommendations using your local LLM",
  resources: ["catalog", "meta", { name: "stream", types: ["movie", "series"], idPrefixes: ["tt"] }],
  types: ["movie", "series"],
  catalogs: [
    { type: "movie", id: "aisearch.top", name: "AI Movie Search", extra: [{ name: "search", isRequired: true }], isSearch: true },
    { type: "series", id: "aisearch.top", name: "AI Series Search", extra: [{ name: "search", isRequired: true }], isSearch: true },
    { type: "movie", id: "aisearch.recommend", name: "AI Movie Recommendations" },
    { type: "series", id: "aisearch.recommend", name: "AI Series Recommendations" },
  ],
  behaviorHints: { configurable: true, configurationRequired: true, searchable: true },
  logo: `${HOST}${BASE_PATH}/logo.png`,
  background: `${HOST}${BASE_PATH}/bg.jpg`,
  contactEmail: "hi@itcon.au",
};

const builder = new addonBuilder(manifest);

function determineIntentFromKeywords(query) {
  if (!query) return "ambiguous";
  const normalizedQuery = query.toLowerCase().trim();
  const movieKeywords = { strong: [/\bmovie(s)?\b/, /\bfilm(s)?\b/, /\bcinema\b/, /\bfeature\b/, /\bmotion picture\b/], medium: [/\bdirector\b/, /\bscreenplay\b/, /\bboxoffice\b/, /\btheater\b/, /\btheatre\b/, /\bcinematic\b/], weak: [/\bwatch\b/, /\bactor\b/, /\bactress\b/, /\bscreenwriter\b/, /\bproducer\b/] };
  const seriesKeywords = { strong: [/\bseries\b/, /\btv show(s)?\b/, /\btelevision\b/, /\bshow(s)?\b/, /\bepisode(s)?\b/, /\bseason(s)?\b/, /\bdocumentary?\b/, /\bdocumentaries?\b/], medium: [/\bnetflix\b/, /\bhbo\b/, /\bhulu\b/, /\bamazon prime\b/, /\bdisney\+\b/, /\bapple tv\+\b/, /\bpilot\b/, /\bfinale\b/], weak: [/\bcharacter\b/, /\bcast\b/, /\bplot\b/, /\bstoryline\b/, /\bnarrative\b/] };
  let movieScore = 0; let seriesScore = 0;
  for (const pattern of movieKeywords.strong) { if (pattern.test(normalizedQuery)) movieScore += 3; }
  for (const pattern of movieKeywords.medium) { if (pattern.test(normalizedQuery)) movieScore += 2; }
  for (const pattern of movieKeywords.weak) { if (pattern.test(normalizedQuery)) movieScore += 1; }
  for (const pattern of seriesKeywords.strong) { if (pattern.test(normalizedQuery)) seriesScore += 3; }
  for (const pattern of seriesKeywords.medium) { if (pattern.test(normalizedQuery)) seriesScore += 2; }
  for (const pattern of seriesKeywords.weak) { if (pattern.test(normalizedQuery)) seriesScore += 1; }
  if (/\b(netflix|hulu|hbo|disney\+|apple tv\+)\b/.test(normalizedQuery)) seriesScore += 1;
  if (/\b(cinema|theatrical|box office|imax)\b/.test(normalizedQuery)) movieScore += 1;
  if (/\b\d{4}-\d{4}\b/.test(normalizedQuery)) seriesScore += 1;
  const scoreDifference = Math.abs(movieScore - seriesScore);
  if (scoreDifference < 2) return "ambiguous";
  return movieScore > seriesScore ? "movie" : "series";
}

function extractGenreCriteria(query) {
  const q = query.toLowerCase();
  const basicGenres = { action: /\b(action)\b/i, comedy: /\b(comedy|comedies|funny)\b/i, drama: /\b(drama|dramas|dramatic)\b/i, horror: /\b(horror|scary|frightening)\b/i, thriller: /\b(thriller|thrillers|suspense)\b/i, romance: /\b(romance|romantic|love)\b/i, scifi: /\b(sci-?fi|science\s*fiction)\b/i, fantasy: /\b(fantasy|magical)\b/i, documentary: /\b(documentary|documentaries)\b/i, animation: /\b(animation|animations|animated|anime)\b/i, adventure: /\b(adventure|adventures)\b/i, crime: /\b(crime|criminal|detective|detectives)\b/i, mystery: /\b(mystery|mysteries|detective|detectives)\b/i, family: /\b(family|kid-friendly|children|childrens)\b/i, biography: /\b(biography|biopic|biographical|biopics)\b/i, history: /\b(history|historical)\b/i, gore: /\b(gore|gory|bloody)\b/i, reality: /\b(reality|realty)\s*(tv|show|series)?\b/i, "talk show": /\b(talk\s*show|talk\s*series)\b/i, soap: /\b(soap\s*opera?|soap\s*series|soap)\b/i, news: /\b(news|newscast|news\s*program)\b/i, kids: /\b(kids?|children|childrens|youth)\b/i };
  const subGenres = { cyberpunk: /\b(cyberpunk|cyber\s*punk)\b/i, noir: /\b(noir|neo-noir)\b/i, psychological: /\b(psychological)\b/i, superhero: /\b(superhero|comic\s*book|marvel|dc)\b/i, musical: /\b(musical|music)\b/i, war: /\b(war|military)\b/i, western: /\b(western|cowboy)\b/i, sports: /\b(sports?|athletic)\b/i };
  const moods = { feelGood: /\b(feel-?good|uplifting|heartwarming)\b/i, dark: /\b(dark|gritty|disturbing)\b/i, thoughtProvoking: /\b(thought-?provoking|philosophical|deep)\b/i, intense: /\b(intense|gripping|edge.*seat)\b/i, lighthearted: /\b(light-?hearted|fun|cheerful)\b/i };
  const supportedGenres = new Set([...Object.keys(basicGenres), ...Object.keys(subGenres)]);
  const genreAliases = { "sci-fi": "scifi", "science fiction": "scifi", "rom-com": "comedy", "romantic comedy": "comedy", "rom com": "comedy", "super hero": "superhero", "super-hero": "superhero" };
  Object.keys(genreAliases).forEach((alias) => { supportedGenres.add(alias); });
  const combinedPattern = /(?:action[- ]comedy|romantic[- ]comedy|sci-?fi[- ]horror|dark[- ]comedy|romantic[- ]thriller)/i;
  const notPattern = /\b(?:not|no|except|excluding)\s+(\w+(?:\s+\w+)?)/gi;
  const excludedGenres = new Set();
  let match;
  while ((match = notPattern.exec(q)) !== null) {
    const negatedTerm = match[1].toLowerCase().trim();
    if (supportedGenres.has(negatedTerm)) { excludedGenres.add(genreAliases[negatedTerm] || negatedTerm); }
    else { for (const [genre, pattern] of Object.entries(basicGenres)) { if (pattern.test(negatedTerm)) { excludedGenres.add(genre); break; } } for (const [genre, pattern] of Object.entries(subGenres)) { if (pattern.test(negatedTerm)) { excludedGenres.add(genre); break; } } }
  }
  const genres = { include: [], exclude: Array.from(excludedGenres), mood: [], style: [] };
  const combinedMatch = q.match(combinedPattern);
  if (combinedMatch) genres.include.push(combinedMatch[0].toLowerCase().replace(/\s+/g, "-"));
  for (const [genre, pattern] of Object.entries(basicGenres)) { if (pattern.test(q) && !excludedGenres.has(genre)) { const genreIndex = q.search(pattern); const beforeGenre = q.substring(0, genreIndex); if (!beforeGenre.match(/\b(not|no|except|excluding)\s+$/)) genres.include.push(genre); } }
  for (const [subgenre, pattern] of Object.entries(subGenres)) { if (pattern.test(q) && !excludedGenres.has(subgenre)) { const genreIndex = q.search(pattern); const beforeGenre = q.substring(0, genreIndex); if (!beforeGenre.match(/\b(not|no|except|excluding)\s+$/)) genres.include.push(subgenre); } }
  for (const [mood, pattern] of Object.entries(moods)) { if (pattern.test(q)) genres.mood.push(mood); }
  return Object.values(genres).some((arr) => arr.length > 0) ? genres : null;
}

function isRecommendationQuery(query) { return query.toLowerCase().trim().startsWith("recommend"); }

function isItemWatchedOrRated(item, watchHistory, ratedItems) {
  if (!item) return false;
  const normalizedName = item.name.toLowerCase().trim();
  const itemYear = parseInt(item.year);
  const isWatched = watchHistory && watchHistory.length > 0 && watchHistory.some((historyItem) => { const media = historyItem.movie || historyItem.show; if (!media) return false; const historyName = media.title.toLowerCase().trim(); const historyYear = parseInt(media.year); return normalizedName === historyName && (!itemYear || !historyYear || itemYear === historyYear); });
  const isRated = ratedItems && ratedItems.length > 0 && ratedItems.some((ratedItem) => { const media = ratedItem.movie || ratedItem.show; if (!media) return false; const ratedName = media.title.toLowerCase().trim(); const ratedYear = parseInt(media.year); return normalizedName === ratedName && (!itemYear || !ratedYear || itemYear === ratedYear); });
  return isWatched || isRated;
}

async function getLandscapeThumbnail(tmdbData, imdbId, fanartApiKey, tmdbKey) {
  if (fanartApiKey && imdbId) {
    try { const fanartThumb = await fetchFanartThumbnail(imdbId, fanartApiKey); if (fanartThumb) return fanartThumb; } catch (error) { logger.debug("Fanart.tv thumbnail fetch failed", { imdbId, error: error.message }); }
  }
  if (tmdbData.backdrop) return tmdbData.backdrop.replace('/original', '/w780');
  return tmdbData.poster;
}

async function fetchFanartThumbnail(imdbId, fanartApiKey) {
  if (!imdbId) return null;
  const effectiveFanartKey = fanartApiKey || DEFAULT_FANART_KEY;
  if (!effectiveFanartKey) return null;
  const cacheKey = `fanart_thumb_${imdbId}`;
  if (fanartCache.has(cacheKey)) return fanartCache.get(cacheKey).data;
  try {
    let FanartApi;
    try { FanartApi = require("fanart.tv"); } catch (requireError) { return null; }
    const fanart = new FanartApi(effectiveFanartKey);
    const data = await withRetry(async () => await fanart.movies.get(imdbId), { maxRetries: 3, baseDelay: 1000, shouldRetry: (error) => !error.status || error.status !== 401, operationName: "Fanart.tv API call" });
    const thumbnail = data?.moviethumb?.filter(thumb => thumb.lang === 'en' || !thumb.lang || thumb.lang.trim() === '')?.sort((a, b) => b.likes - a.likes)[0]?.url;
    fanartCache.set(cacheKey, { timestamp: Date.now(), data: thumbnail });
    return thumbnail;
  } catch (error) {
    fanartCache.set(cacheKey, { timestamp: Date.now(), data: null });
    return null;
  }
}

async function fetchRpdbPoster(imdbId, rpdbKey, posterType = "poster-default", isTier0User = false) {
  if (!imdbId || !rpdbKey) return null;
  const cacheKey = `rpdb_${imdbId}_${posterType}`;
  if (isTier0User && rpdbCache.has(cacheKey)) return rpdbCache.get(cacheKey).data;
  try {
    const url = `https://api.ratingposterdb.com/${rpdbKey}/imdb/${posterType}/${imdbId}.jpg`;
    const posterUrl = await withRetry(async () => {
      const response = await fetch(url);
      if (response.status === 404) { if (isTier0User) rpdbCache.set(cacheKey, { timestamp: Date.now(), data: null }); return null; }
      if (!response.ok) { const error = new Error(`RPDB API error: ${response.status}`); error.status = response.status; throw error; }
      return url;
    }, { maxRetries: 2, initialDelay: 1000, maxDelay: 5000, shouldRetry: (error) => error.status !== 404 && (!error.status || error.status >= 500), operationName: "RPDB poster API call" });
    if (isTier0User) rpdbCache.set(cacheKey, { timestamp: Date.now(), data: posterUrl });
    return posterUrl;
  } catch (error) {
    logger.error("RPDB API Error:", { error: error.message, imdbId, posterType });
    return null;
  }
}

function getRpdbTierFromApiKey(apiKey) {
  if (!apiKey) return -1;
  try { const tierMatch = apiKey.match(/^t(\d+)-/); if (tierMatch && tierMatch[1] !== undefined) return parseInt(tierMatch[1]); return -1; } catch (error) { return -1; }
}

async function toStremioMeta(item, platform = "unknown", tmdbKey, rpdbKey, rpdbPosterType = "poster-default", language = "en-US", config, includeAdult = false) {
  if (!item.id || !item.name) return null;
  const type = item.type || (item.id.includes("movie") ? "movie" : "series");
  const enableRpdb = config?.EnableRpdb !== undefined ? config.EnableRpdb : false;
  const userRpdbKey = config?.RpdbApiKey;
  const usingUserKey = !!userRpdbKey;
  const usingDefaultKey = !userRpdbKey && !!DEFAULT_RPDB_KEY;
  const userTier = usingUserKey ? getRpdbTierFromApiKey(userRpdbKey) : -1;
  const isTier0User = (usingUserKey && userTier === 0) || usingDefaultKey;
  const tmdbData = await searchTMDB(item.name, type, item.year, tmdbKey, language, includeAdult);
  if (!tmdbData || !tmdbData.imdb_id) return null;
  let poster = tmdbData.poster;
  const effectiveRpdbKey = userRpdbKey || DEFAULT_RPDB_KEY;
  if (enableRpdb && effectiveRpdbKey && tmdbData.imdb_id) {
    try { const rpdbPoster = await fetchRpdbPoster(tmdbData.imdb_id, effectiveRpdbKey, rpdbPosterType, isTier0User); if (rpdbPoster) poster = rpdbPoster; } catch (error) { logger.debug("RPDB poster fetch failed", { error: error.message }); }
  }
  if (!poster) return null;
  const meta = { id: tmdbData.imdb_id, type: type, name: tmdbData.title || tmdbData.name, description: platform === "android-tv" ? (tmdbData.overview || "").slice(0, 200) : tmdbData.overview || "", year: parseInt(item.year) || 0, poster: platform === "android-tv" && poster.includes("/w500/") ? poster.replace("/w500/", "/w342/") : poster, background: tmdbData.backdrop, posterShape: "regular" };
  if (tmdbData.genres && tmdbData.genres.length > 0) meta.genres = tmdbData.genres.map((id) => (type === "series" ? TMDB_TV_GENRES[id] : TMDB_GENRES[id])).filter(Boolean);
  return meta;
}

function detectPlatform(extra = {}) {
  if (extra.headers?.["stremio-platform"]) return extra.headers["stremio-platform"];
  const userAgent = (extra.userAgent || extra.headers?.["stremio-user-agent"] || "").toLowerCase();
  if (userAgent.includes("android tv") || userAgent.includes("chromecast") || userAgent.includes("androidtv")) return "android-tv";
  if (userAgent.includes("android") || userAgent.includes("mobile") || userAgent.includes("phone")) return "mobile";
  if (userAgent.includes("windows") || userAgent.includes("macintosh") || userAgent.includes("linux")) return "desktop";
  return "unknown";
}

async function getTmdbDetailsByImdbId(imdbId, type, tmdbKey, language = "en-US") {
  const cacheKey = `details_imdb_${imdbId}_${type}_${language}`;
  if (tmdbDetailsCache.has(cacheKey)) return tmdbDetailsCache.get(cacheKey).data;
  try {
    const findUrl = `${TMDB_API_BASE}/find/${imdbId}?api_key=${tmdbKey}&language=${language}&external_source=imdb_id`;
    const response = await fetch(findUrl);
    if (!response.ok) throw new Error(`TMDB find API error: ${response.status}`);
    const data = await response.json();
    const results = type === 'movie' ? data.movie_results : data.tv_results;
    if (results && results.length > 0) { const details = results[0]; tmdbDetailsCache.set(cacheKey, { timestamp: Date.now(), data: details }); return details; }
    return null;
  } catch (error) { logger.error("Error fetching TMDB details by IMDB ID", { imdbId, error: error.message }); return null; }
}

const TMDB_GENRES = { 28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western" };
const TMDB_TV_GENRES = { 10759: "Action & Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family", 10762: "Kids", 9648: "Mystery", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics", 37: "Western" };

function clearTmdbCache() { const size = tmdbCache.size; tmdbCache.clear(); return { cleared: true, previousSize: size }; }
function clearTmdbDetailsCache() { const size = tmdbDetailsCache.size; tmdbDetailsCache.clear(); return { cleared: true, previousSize: size }; }
function clearTmdbDiscoverCache() { const size = tmdbDiscoverCache.size; tmdbDiscoverCache.clear(); return { cleared: true, previousSize: size }; }
function removeTmdbDiscoverCacheItem(cacheKey) { if (!cacheKey) return { success: false, message: "No cache key provided" }; if (!tmdbDiscoverCache.has(cacheKey)) return { success: false, message: "Cache key not found", key: cacheKey }; tmdbDiscoverCache.delete(cacheKey); return { success: true, message: "Cache item removed successfully", key: cacheKey }; }
function listTmdbDiscoverCacheKeys() { const keys = Array.from(tmdbDiscoverCache.cache.keys()); return { success: true, count: keys.length, keys: keys }; }
function clearAiCache() { const size = aiRecommendationsCache.size; aiRecommendationsCache.clear(); return { cleared: true, previousSize: size }; }
function removeAiCacheByKeywords(keywords) {
  if (!keywords || typeof keywords !== "string") throw new Error("Invalid keywords parameter");
  const searchPhrase = keywords.toLowerCase().trim();
  const removedEntries = [];
  let totalRemoved = 0;
  for (const key of aiRecommendationsCache.keys()) {
    const query = key.split("_")[0].toLowerCase();
    if (query.includes(searchPhrase)) { const entry = aiRecommendationsCache.get(key); if (entry) { removedEntries.push({ key, timestamp: new Date(entry.timestamp).toISOString(), query: key.split("_")[0] }); aiRecommendationsCache.delete(key); totalRemoved++; } }
  }
  return { removed: totalRemoved, entries: removedEntries };
}
function clearRpdbCache() { const size = rpdbCache.size; rpdbCache.clear(); return { cleared: true, previousSize: size }; }
function clearFanartCache() { const size = fanartCache.size; fanartCache.clear(); return { cleared: true, previousSize: size }; }
function clearTraktCache() { const size = traktCache.size; traktCache.clear(); return { cleared: true, previousSize: size }; }
function clearTraktRawDataCache() { const size = traktRawDataCache.size; traktRawDataCache.clear(); return { cleared: true, previousSize: size }; }
function clearQueryAnalysisCache() { const size = queryAnalysisCache.size; queryAnalysisCache.clear(); return { cleared: true, previousSize: size }; }
function clearSimilarContentCache() { const size = similarContentCache.size; similarContentCache.clear(); return { cleared: true, previousSize: size }; }

function getCacheStats() {
  return {
    tmdbCache: { size: tmdbCache.size, maxSize: tmdbCache.max, usagePercentage: ((tmdbCache.size / tmdbCache.max) * 100).toFixed(2) + "%" },
    tmdbDetailsCache: { size: tmdbDetailsCache.size, maxSize: tmdbDetailsCache.max, usagePercentage: ((tmdbDetailsCache.size / tmdbDetailsCache.max) * 100).toFixed(2) + "%" },
    tmdbDiscoverCache: { size: tmdbDiscoverCache.size, maxSize: tmdbDiscoverCache.max, usagePercentage: ((tmdbDiscoverCache.size / tmdbDiscoverCache.max) * 100).toFixed(2) + "%" },
    aiCache: { size: aiRecommendationsCache.size, maxSize: aiRecommendationsCache.max, usagePercentage: ((aiRecommendationsCache.size / aiRecommendationsCache.max) * 100).toFixed(2) + "%" },
    rpdbCache: { size: rpdbCache.size, maxSize: rpdbCache.max, usagePercentage: ((rpdbCache.size / rpdbCache.max) * 100).toFixed(2) + "%" },
    fanartCache: { size: fanartCache.size, maxSize: fanartCache.max, usagePercentage: ((fanartCache.size / fanartCache.max) * 100).toFixed(2) + "%" },
    traktCache: { size: traktCache.size, maxSize: traktCache.max, usagePercentage: ((traktCache.size / traktCache.max) * 100).toFixed(2) + "%" },
    traktRawDataCache: { size: traktRawDataCache.size, maxSize: traktRawDataCache.max, usagePercentage: ((traktRawDataCache.size / traktRawDataCache.max) * 100).toFixed(2) + "%" },
    queryAnalysisCache: { size: queryAnalysisCache.size, maxSize: queryAnalysisCache.max, usagePercentage: ((queryAnalysisCache.size / queryAnalysisCache.max) * 100).toFixed(2) + "%" },
    similarContentCache: { size: similarContentCache.size, maxSize: similarContentCache.max, usagePercentage: ((similarContentCache.size / similarContentCache.max) * 100).toFixed(2) + "%" },
  };
}

function serializeAllCaches() {
  return { tmdbCache: tmdbCache.serialize(), tmdbDetailsCache: tmdbDetailsCache.serialize(), tmdbDiscoverCache: tmdbDiscoverCache.serialize(), aiRecommendationsCache: aiRecommendationsCache.serialize(), rpdbCache: rpdbCache.serialize(), fanartCache: fanartCache.serialize(), traktCache: traktCache.serialize(), traktRawDataCache: traktRawDataCache.serialize(), queryAnalysisCache: queryAnalysisCache.serialize(), similarContentCache: similarContentCache.serialize(), stats: { queryCounter } };
}

function deserializeAllCaches(data) {
  const results = {};
  if (data.tmdbCache) results.tmdbCache = tmdbCache.deserialize(data.tmdbCache);
  if (data.tmdbDetailsCache) results.tmdbDetailsCache = tmdbDetailsCache.deserialize(data.tmdbDetailsCache);
  if (data.tmdbDiscoverCache) results.tmdbDiscoverCache = tmdbDiscoverCache.deserialize(data.tmdbDiscoverCache);
  if (data.aiRecommendationsCache) results.aiRecommendationsCache = aiRecommendationsCache.deserialize(data.aiRecommendationsCache);
  else if (data.aiCache) results.aiRecommendationsCache = aiRecommendationsCache.deserialize(data.aiCache);
  if (data.rpdbCache) results.rpdbCache = rpdbCache.deserialize(data.rpdbCache);
  if (data.fanartCache) results.fanartCache = fanartCache.deserialize(data.fanartCache);
  if (data.traktCache) results.traktCache = traktCache.deserialize(data.traktCache);
  if (data.traktRawDataCache) results.traktRawDataCache = traktRawDataCache.deserialize(data.traktRawDataCache);
  if (data.queryAnalysisCache) results.queryAnalysisCache = queryAnalysisCache.deserialize(data.queryAnalysisCache);
  if (data.similarContentCache) results.similarContentCache = similarContentCache.deserialize(data.similarContentCache);
  if (data.stats && typeof data.stats.queryCounter === "number") queryCounter = data.stats.queryCounter;
  return results;
}

/**
 * Uses local LLM to discover content type and genres for a query
 */
async function discoverTypeAndGenres(query, llmBaseUrl, llmModel, llmApiKey) {
  const promptText = `Analyze this recommendation query: "${query}"

Determine:
1. What type of content is being requested (movie, series, or ambiguous)
2. What genres are relevant to this query

Respond in a single line with pipe-separated format:
type|genre1,genre2,genre3

Where type is one of: movie, series, ambiguous
Genres are comma-separated, or "all" if no specific genres apply.

Examples:
movie|action,thriller,sci-fi
series|comedy,drama
ambiguous|romance,comedy
movie|all

Do not include any explanatory text. Just the single line.`;

  try {
    const text = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);
    const firstLine = text.split("\n")[0].trim();
    const parts = firstLine.split("|");
    if (parts.length !== 2) return { type: "ambiguous", genres: [] };
    let type = parts[0].trim().toLowerCase();
    if (type !== "movie" && type !== "series") type = "ambiguous";
    const genres = parts[1].split(",").map((g) => g.trim()).filter((g) => g.length > 0 && g.toLowerCase() !== "ambiguous");
    if (genres.length === 1 && genres[0].toLowerCase() === "all") return { type, genres: [] };
    return { type, genres };
  } catch (error) {
    logger.error("Genre discovery LLM error", { error: error.message });
    return { type: "ambiguous", genres: [] };
  }
}

function filterTraktDataByGenres(traktData, genres) {
  if (!traktData || !genres || genres.length === 0) return { recentlyWatched: [], highlyRated: [], lowRated: [] };
  const { watched, rated } = traktData;
  const genreSet = new Set(genres.map((g) => g.toLowerCase()));
  const hasMatchingGenre = (item) => { const media = item.movie || item.show; if (!media || !media.genres || media.genres.length === 0) return false; return media.genres.some((g) => genreSet.has(g.toLowerCase())); };
  return { recentlyWatched: (watched || []).filter(hasMatchingGenre).slice(0, 100), highlyRated: (rated || []).filter((item) => item.rating >= 4).filter(hasMatchingGenre).slice(0, 100), lowRated: (rated || []).filter((item) => item.rating <= 2).filter(hasMatchingGenre).slice(0, 100) };
}

function incrementQueryCounter() { queryCounter++; return queryCounter; }
function getQueryCount() { return queryCounter; }
function setQueryCount(newCount) { if (typeof newCount !== "number" || newCount < 0) throw new Error("Query count must be a non-negative number"); queryCounter = newCount; return queryCounter; }

const catalogHandler = async function (args, req) {
  const startTime = Date.now();
  const { id, type, extra } = args;
  let isHomepageQuery = false;

  try {
    const configData = args.config;
    if (!configData || Object.keys(configData).length === 0) {
      const errorMeta = createErrorMeta('Configuration Missing', 'The addon has not been configured yet. Please set your API keys.');
      return { metas: [errorMeta] };
    }

    const tmdbKey = configData.TmdbApiKey;
    // Use user-configured LLM settings or fall back to server environment defaults
    const llmBaseUrl = configData.LlmBaseUrl || DEFAULT_LLM_BASE_URL;
    const llmModel = configData.LlmModel || DEFAULT_LLM_MODEL;
    const llmApiKey = configData.LlmApiKey || DEFAULT_LLM_API_KEY;

    if (configData.traktConnectionError) {
      const errorMeta = createErrorMeta('Trakt Connection Failed', 'Your access to Trakt.tv has expired or was revoked. Please log in again via the addon configuration page.');
      return { metas: [errorMeta] };
    }
    if (!tmdbKey || tmdbKey.length < 10) {
      const errorMeta = createErrorMeta('TMDB API Key Invalid', 'Your TMDB API key is missing or invalid. Please correct it in the addon settings.');
      return { metas: [errorMeta] };
    }
    const tmdbValidationUrl = `https://api.themoviedb.org/3/configuration?api_key=${tmdbKey}`;
    const tmdbResponse = await fetch(tmdbValidationUrl);
    if (!tmdbResponse.ok) {
      const errorMeta = createErrorMeta('TMDB API Key Invalid', `The key failed validation (Status: ${tmdbResponse.status}). Please check your TMDB key.`);
      return { metas: [errorMeta] };
    }
    if (!llmBaseUrl) {
      const errorMeta = createErrorMeta('LLM Not Configured', 'No LLM base URL is set. Please configure LlmBaseUrl in your addon settings or set LLM_BASE_URL on the server.');
      return { metas: [errorMeta] };
    }

    let searchQuery = "";
    if (typeof extra === "string" && extra.includes("search=")) searchQuery = decodeURIComponent(extra.split("search=")[1]);
    else if (extra?.search) searchQuery = extra.search;

    if (!searchQuery) {
      if (id.startsWith("aisearch.home.")) {
        isHomepageQuery = true;
        let homepageQueries = configData.HomepageQuery;
        if (!homepageQueries || homepageQueries.trim() === '') homepageQueries = "AI Recommendations:recommend a hidden gem movie, AI Recommendations:recommend a binge-worthy series";
        const idParts = id.split(".");
        if (idParts.length === 4 && homepageQueries) {
          const queryIndex = parseInt(idParts[2], 10);
          const catalogEntries = homepageQueries.split(",").map(q => q.trim());
          if (!isNaN(queryIndex) && catalogEntries[queryIndex]) {
            const entry = catalogEntries[queryIndex];
            const parts = entry.split(/:(.*)/s);
            if (parts.length > 1 && parts[1].trim()) searchQuery = parts[1].trim();
            else searchQuery = entry;
          }
        }
        if (!searchQuery) { const errorMeta = createErrorMeta('Configuration Error', 'Could not find the matching homepage query for this catalog.'); return { metas: [errorMeta] }; }
      } else {
        const errorMeta = createErrorMeta('Search Required', 'Please enter a search term to get AI recommendations.');
        return { metas: [errorMeta] };
      }
    }

    const language = configData.TmdbLanguage || "en-US";
    const rpdbKey = configData.RpdbApiKey || DEFAULT_RPDB_KEY;
    const rpdbPosterType = configData.RpdbPosterType || "poster-default";
    let numResults = parseInt(configData.NumResults) || 20;
    if (numResults > 30) numResults = MAX_AI_RECOMMENDATIONS;
    const enableAiCache = configData.EnableAiCache !== undefined ? configData.EnableAiCache : true;
    const enableRpdb = configData.EnableRpdb !== undefined ? configData.EnableRpdb : false;
    const includeAdult = configData.IncludeAdult === true;
    const platform = detectPlatform(extra);
    const isSearchRequest = (typeof extra === "string" && extra.includes("search=")) || !!extra?.search;
    if (isSearchRequest) logger.info("Processing search query", { searchQuery, type });

    const intent = determineIntentFromKeywords(searchQuery);
    if (intent !== "ambiguous" && intent !== type) return { metas: [] };

    let exactMatchMeta = null;
    let tmdbInitialResults = [];
    let matchResult = null;

    if (!isRecommendationQuery(searchQuery)) {
      matchResult = await searchTMDBExactMatch(searchQuery, type, tmdbKey, language, includeAdult);
      if (matchResult) {
        tmdbInitialResults = matchResult.results;
        if (matchResult.isExactMatch) {
          const normalizedTitle = searchQuery.toLowerCase().trim();
          const exactMatchData = matchResult.results.find(r => (r.title || r.name || "").toLowerCase().trim() === normalizedTitle);
          if (exactMatchData) {
            const details = await getTmdbDetailsByImdbId(exactMatchData.id, type, tmdbKey);
            if (details && details.imdb_id) {
              const exactMatchItem = { id: `exact_${exactMatchData.id}`, name: exactMatchData.title || exactMatchData.name, year: (exactMatchData.release_date || exactMatchData.first_air_date || 'N/A').substring(0, 4), type: type };
              exactMatchMeta = await toStremioMeta(exactMatchItem, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData, includeAdult);
            }
          }
        }
      }
    }

    const isRecommendation = isRecommendationQuery(searchQuery);
    let discoveredGenres = [];
    let traktData = null;
    let filteredTraktData = null;

    if (isRecommendation) {
      const discoveryResult = await discoverTypeAndGenres(searchQuery, llmBaseUrl, llmModel, llmApiKey);
      discoveredGenres = discoveryResult.genres;

      if (DEFAULT_TRAKT_CLIENT_ID && configData.TraktAccessToken) {
        traktData = await fetchTraktWatchedAndRated(DEFAULT_TRAKT_CLIENT_ID, configData.TraktAccessToken, type === "movie" ? "movies" : "shows", configData);
        if (traktData) {
          if (discoveredGenres.length > 0) {
            filteredTraktData = filterTraktDataByGenres(traktData, discoveredGenres);
          } else {
            filteredTraktData = { recentlyWatched: traktData.watched?.slice(0, 100) || [], highlyRated: (traktData.rated || []).filter((item) => item.rating >= 4).slice(0, 25), lowRated: (traktData.rated || []).filter((item) => item.rating <= 2).slice(0, 25) };
          }
        }
      }
    }

    const cacheKey = `${searchQuery}_${type}_${traktData ? "trakt" : "no_trakt"}`;
    if (enableAiCache && !traktData && !isHomepageQuery && aiRecommendationsCache.has(cacheKey)) {
      const cached = aiRecommendationsCache.get(cacheKey);
      if (cached.configNumResults && numResults > cached.configNumResults) {
        aiRecommendationsCache.delete(cacheKey);
      } else if (!cached.data?.recommendations || (type === "movie" && !cached.data.recommendations.movies) || (type === "series" && !cached.data.recommendations.series)) {
        aiRecommendationsCache.delete(cacheKey);
      } else {
        const selectedRecommendations = type === "movie" ? cached.data.recommendations.movies || [] : cached.data.recommendations.series || [];
        if (selectedRecommendations.length === 0) { const errorMeta = createErrorMeta('No Results Found', 'The AI could not find any recommendations for your query.'); return { metas: [errorMeta] }; }
        const metas = (await Promise.all(selectedRecommendations.map((item) => toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData, includeAdult)))).filter(Boolean);
        if (metas.length === 0 && !exactMatchMeta) { const errorMeta = createErrorMeta('Data Fetch Error', 'Could not retrieve details for any of the AI recommendations.'); return { metas: [errorMeta] }; }
        let finalMetas = metas;
        if (exactMatchMeta) finalMetas = [exactMatchMeta, ...metas.filter((meta) => meta.id !== exactMatchMeta.id)];
        if (finalMetas.length === 0) { const errorMeta = createErrorMeta('No Results Found', 'The AI could not find any recommendations for your query.'); return { metas: [errorMeta] }; }
        if (finalMetas.length > 0 && isSearchRequest) incrementQueryCounter();
        return { metas: finalMetas };
      }
    }

    // Build the prompt for local LLM
    try {
      const genreCriteria = extractGenreCriteria(searchQuery);
      const currentYear = new Date().getFullYear();
      let franchiseInstruction = `your TOP PRIORITY is to list ALL official mainline movies from that franchise, followed by any relevant spin-offs or related content.`;
      if (type === 'series') franchiseInstruction = `your TOP PRIORITY is to provide a comprehensive list of ALL television content related to that franchise, including narrative series, documentaries, reality shows, and TV specials.`;

      let promptLines = [
        `You are a ${type} recommendation expert. Analyze this query: "${searchQuery}"`,
        "",
        "QUERY ANALYSIS:",
      ];
      if (isRecommendation && discoveredGenres.length > 0) promptLines.push(`Discovered genres: ${discoveredGenres.join(", ")}`);
      else if (genreCriteria?.include?.length > 0) promptLines.push(`Requested genres: ${genreCriteria.include.join(", ")}`);
      if (genreCriteria?.mood?.length > 0) promptLines.push(`Mood/Style: ${genreCriteria.mood.join(", ")}`);
      promptLines.push("");

      if (traktData && isRecommendation) {
        const { preferences } = traktData;
        const { recentlyWatched, highlyRated, lowRated } = filteredTraktData || { recentlyWatched: traktData.watched?.slice(0, 100) || [], highlyRated: (traktData.rated || []).filter((item) => item.rating >= 4).slice(0, 25), lowRated: (traktData.rated || []).filter((item) => item.rating <= 2).slice(0, 25) };
        promptLines.push("USER'S WATCH HISTORY AND PREFERENCES (FILTERED BY RELEVANT GENRES):", "");
        if (recentlyWatched.length > 0) promptLines.push("Recently watched in these genres:", recentlyWatched.slice(0, 25).map((item) => { const media = item.movie || item.show; return `- ${media.title} (${media.year}) - ${media.genres?.join(", ") || "N/A"}`; }).join("\n"), "");
        if (highlyRated.length > 0) promptLines.push("Highly rated (4-5 stars):", highlyRated.slice(0, 25).map((item) => { const media = item.movie || item.show; return `- ${media.title} (${item.rating}/5) - ${media.genres?.join(", ") || "N/A"}`; }).join("\n"), "");
        if (lowRated.length > 0) promptLines.push("Low rated (1-2 stars):", lowRated.slice(0, 15).map((item) => { const media = item.movie || item.show; return `- ${media.title} (${item.rating}/5)`; }).join("\n"), "");
        if (discoveredGenres.length === 0) promptLines.push("Top genres:", preferences.genres.map((g) => `- ${g.genre} (Score: ${g.count.toFixed(2)})`).join("\n"), "");
        promptLines.push("Favorite actors:", preferences.actors.map((a) => `- ${a.actor}`).join("\n"), "", "Preferred directors:", preferences.directors.map((d) => `- ${d.director}`).join("\n"), "");
      }

      if (tmdbInitialResults.length > 0) {
        const initialTitles = tmdbInitialResults.slice(0, 15).map(item => `- ${item.title || item.name} (${(item.release_date || item.first_air_date || 'N/A').substring(0, 4)})`).join('\n');
        promptLines.push("CONTEXT FROM INITIAL DATABASE SEARCH:", "Use this as primary data source, add similar titles, and sort by relevance:", "", initialTitles, "");
      }

      const examplesText = type === 'movie' ? "EXAMPLES:\nmovie|The Matrix|1999\nmovie|Inception|2010" : "EXAMPLES:\nseries|Breaking Bad|2008\nseries|Game of Thrones|2011";

      promptLines = promptLines.concat([
        "IMPORTANT INSTRUCTIONS:",
        `- Current year is ${currentYear}.`,
        `- 'recent' means within the last 2-3 years (${currentYear - 2} to ${currentYear})`,
        `- 'new' or 'latest' means released in ${currentYear}`,
        "SPECIFIC QUERY HANDLING:",
        `1. FRANCHISE/SERIES: If the query is for a specific franchise, ${franchiseInstruction} List them in strict chronological order.`,
        `2. ACTOR/DIRECTOR FILMOGRAPHY: List their most notable works chronologically.`,
        `3. GENERAL RECOMMENDATIONS: Provide diverse recommendations matching the query's theme and genre, ordered by relevance.`,
        "CRITICAL REQUIREMENTS:",
      ]);
      if (traktData) { promptLines.push(`- DO NOT recommend any content that appears in the user's watch history or ratings above.`, `- Recommend content SIMILAR to the user's highly rated content but NOT THE SAME ones.`); }
      promptLines = promptLines.concat([
        `- You MUST return up to ${numResults} ${type} recommendations.`,
        `- Prioritize quality over exact matching.`,
        "",
        "RESPONSE FORMAT (no additional commentary):",
        "[type]|[name]|[year]",
        "",
        examplesText,
        "",
        "RULES:",
        "- Use | separator",
        "- Year: YYYY format",
        `- Type: accurately label each item as 'movie' or 'series'`,
        "- Titles: clean official titles only, no extra descriptions",
        "- Only include officially released movies and TV series",
      ]);
      if (genreCriteria) {
        if (genreCriteria.include.length > 0) promptLines.push(`- Must match genres: ${genreCriteria.include.join(", ")}`);
        if (genreCriteria.exclude.length > 0) promptLines.push(`- Exclude genres: ${genreCriteria.exclude.join(", ")}`);
        if (genreCriteria.mood.length > 0) promptLines.push(`- Match mood/style: ${genreCriteria.mood.join(", ")}`);
      }

      const promptText = promptLines.join("\n");
      logger.info("Making local LLM API call", { model: llmModel, baseUrl: llmBaseUrl, query: searchQuery, type });

      const text = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);

      const lines = text.split("\n").map((line) => line.trim()).filter((line) => line && !line.startsWith("type|"));
      const recommendations = { movies: type === "movie" ? [] : undefined, series: type === "series" ? [] : undefined };
      let validRecommendations = 0;
      let invalidLines = 0;

      for (const line of lines) {
        try {
          const parts = line.split("|");
          let lineType, name, year;
          if (parts.length === 3) { [lineType, name, year] = parts.map((s) => s.trim()); }
          else if (parts.length === 2) {
            lineType = parts[0].trim();
            const nameWithYear = parts[1].trim();
            const yearMatch = nameWithYear.match(/\((\d{4})\)$/);
            if (yearMatch) { year = yearMatch[1]; name = nameWithYear.substring(0, nameWithYear.lastIndexOf("(")).trim(); }
            else { const anyYearMatch = nameWithYear.match(/\b(19\d{2}|20\d{2})\b/); if (anyYearMatch) { year = anyYearMatch[0]; name = nameWithYear.replace(anyYearMatch[0], "").trim(); } else { invalidLines++; continue; } }
          } else { invalidLines++; continue; }
          const yearNum = parseInt(year);
          if (!lineType || !name || isNaN(yearNum)) { invalidLines++; continue; }
          if (lineType === type && name && yearNum) {
            const item = { name, year: yearNum, type, id: `ai_${type}_${name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}` };
            if (type === "movie") recommendations.movies.push(item);
            else if (type === "series") recommendations.series.push(item);
            validRecommendations++;
          }
        } catch (error) { invalidLines++; }
      }

      const finalResult = { recommendations, fromCache: false };

      if (traktData && isRecommendation) {
        const watchHistory = traktData.watched.concat(traktData.history || []);
        if (finalResult.recommendations.movies) finalResult.recommendations.movies = finalResult.recommendations.movies.filter((movie) => !isItemWatchedOrRated(movie, watchHistory, traktData.rated));
        if (finalResult.recommendations.series) finalResult.recommendations.series = finalResult.recommendations.series.filter((series) => !isItemWatchedOrRated(series, watchHistory, traktData.rated));
      }

      const recommendationsToCache = finalResult.recommendations;
      const hasMoviesToCache = recommendationsToCache.movies && recommendationsToCache.movies.length > 0;
      const hasSeriesToCache = recommendationsToCache.series && recommendationsToCache.series.length > 0;
      if ((hasMoviesToCache || hasSeriesToCache) && !traktData && !isHomepageQuery && enableAiCache) {
        aiRecommendationsCache.set(cacheKey, { timestamp: Date.now(), data: finalResult, configNumResults: numResults });
      }

      const selectedRecommendations = type === "movie" ? finalResult.recommendations.movies || [] : finalResult.recommendations.series || [];
      const metas = (await Promise.all(selectedRecommendations.map((item) => toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData)))).filter(Boolean);

      let finalMetas = metas;
      if (exactMatchMeta) finalMetas = [exactMatchMeta, ...metas.filter((meta) => meta.id !== exactMatchMeta.id)];
      if (finalMetas.length === 0) { const errorMeta = createErrorMeta('No Results Found', 'The AI could not find any recommendations for your query. Please try rephrasing your search.'); return { metas: [errorMeta] }; }
      if (finalMetas.length > 0 && isSearchRequest) incrementQueryCounter();
      return { metas: finalMetas };
    } catch (error) {
      logger.error("Local LLM Error:", { error: error.message, stack: error.stack, query: searchQuery });
      let errorMessage = 'The local LLM failed to respond. Check that your LLM server is running.';
      if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) errorMessage = `Could not connect to your LLM server at ${args.config?.LlmBaseUrl || DEFAULT_LLM_BASE_URL}. Is it running?`;
      const errorMeta = createErrorMeta('LLM Error', errorMessage);
      return { metas: [errorMeta] };
    }
  } catch (error) {
    logger.error("Catalog processing error", { error: error.message, stack: error.stack });
    const errorMeta = createErrorMeta('Addon Error', 'A critical error occurred. Please check the server logs.');
    return { metas: [errorMeta] };
  }
};

const streamHandler = async (args, req) => {
  const { config } = args;
  if (config) {
    try {
      const decryptedConfigStr = decryptConfig(config);
      if (decryptedConfigStr) {
        const configData = JSON.parse(decryptedConfigStr);
        const enableSimilar = configData.EnableSimilar !== undefined ? configData.EnableSimilar : true;
        if (!enableSimilar) return Promise.resolve({ streams: [] });
      }
    } catch (error) { logger.error("Failed to read 'EnableSimilar' config in streamHandler", { error: error.message }); }
  }
  const isWeb = req.headers["origin"]?.includes("web.stremio.com");
  const stremioUrlPrefix = isWeb ? "https://web.stremio.com/#" : "stremio://";
  const stream = { name: "✨ AI Search", description: "Similar movies and shows.", externalUrl: `${stremioUrlPrefix}/detail/${args.type}/ai-recs:${args.id}`, behaviorHints: { notWebReady: true } };
  return Promise.resolve({ streams: [stream] });
};

const metaHandler = async function (args) {
  const { type, id, config } = args;
  const startTime = Date.now();
  try {
    if (!id || !id.startsWith('ai-recs:')) return { meta: null };
    if (config) {
      const decryptedConfigStr = decryptConfig(config);
      if (!decryptedConfigStr) throw new Error("Failed to decrypt config data in metaHandler");
      const configData = JSON.parse(decryptedConfigStr);
      const { TmdbApiKey, GeminiModel, NumResults, RpdbApiKey, RpdbPosterType, TmdbLanguage, FanartApiKey } = configData;
      const llmBaseUrl = configData.LlmBaseUrl || DEFAULT_LLM_BASE_URL;
      const llmModel = configData.LlmModel || DEFAULT_LLM_MODEL;
      const llmApiKey = configData.LlmApiKey || DEFAULT_LLM_API_KEY;
      const originalId = id.split(':')[1];
      const cacheKey = `similar_${originalId}_${type}_${NumResults || 15}`;
      const cached = similarContentCache.get(cacheKey);
      if (cached) return { meta: cached.data };
      let sourceDetails = await getTmdbDetailsByImdbId(originalId, type, TmdbApiKey);
      if (!sourceDetails) { const fallbackType = type === 'movie' ? 'series' : 'movie'; sourceDetails = await getTmdbDetailsByImdbId(originalId, fallbackType, TmdbApiKey); }
      if (!sourceDetails) throw new Error(`Could not find source details for original ID: ${originalId}`);
      const sourceTitle = sourceDetails.title || sourceDetails.name;
      const sourceYear = (sourceDetails.release_date || sourceDetails.first_air_date || "").substring(0, 4);
      let numResults = parseInt(NumResults) || 15;
      if (numResults > 25) numResults = 25;

      const promptText = `You are an expert recommendation engine for movies and TV shows.
Generate exactly ${numResults} recommendations similar to "${sourceTitle} (${sourceYear})".

PART 1: List all other official movies/series from the same franchise as "${sourceTitle}", sorted chronologically.
PART 2: Fill remaining slots with highly similar unrelated titles, sorted by relevance.

CRITICAL: Do NOT include "${sourceTitle} (${sourceYear})" itself. No headers or explanations.

Format (one per line):
type|name|year

Example:
movie|Batman Begins|2005
movie|The Dark Knight Rises|2012`;

      const responseText = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);
      const lines = responseText.split('\n').map(line => line.trim()).filter(Boolean);
      const videoPromises = lines.map(async (line) => {
        const parts = line.split('|');
        if (parts.length !== 3) return null;
        const [recType, name, year] = parts.map(p => p.trim());
        const tmdbData = await searchTMDB(name, recType, year, TmdbApiKey);
        if (tmdbData && tmdbData.imdb_id) {
          let description = tmdbData.overview || "";
          if (tmdbData.tmdbRating && tmdbData.tmdbRating > 0) description = `⭐ TMDB Rating: ${tmdbData.tmdbRating.toFixed(1)}/10\n\n${description}`;
          const landscapeThumbnail = await getLandscapeThumbnail(tmdbData, tmdbData.imdb_id, FanartApiKey, TmdbApiKey);
          return { id: tmdbData.imdb_id, title: tmdbData.title, released: new Date(tmdbData.release_date || '1970-01-01').toISOString(), overview: description, thumbnail: landscapeThumbnail };
        }
        return null;
      });
      const videos = (await Promise.all(videoPromises)).filter(Boolean);
      const meta = { id: id, type: 'series', name: `AI: Recommendations for ${sourceTitle}`, description: `A collection of titles similar to ${sourceTitle} (${sourceYear}), generated by AI.`, poster: sourceDetails.poster_path ? `https://image.tmdb.org/t/p/w500${sourceDetails.poster_path}` : null, background: sourceDetails.backdrop_path ? `https://image.tmdb.org/t/p/original${sourceDetails.backdrop_path}` : null, videos: videos };
      if (videos.length > 0) similarContentCache.set(cacheKey, { timestamp: Date.now(), data: meta });
      return { meta };
    }
  } catch (error) { logger.error("Meta Handler Error:", { message: error.message, stack: error.stack, id: id }); }
  return { meta: null };
};

builder.defineCatalogHandler(catalogHandler);
builder.defineStreamHandler(streamHandler);
builder.defineMetaHandler(metaHandler);

const addonInterface = builder.getInterface();

module.exports = {
  builder, addonInterface, catalogHandler, streamHandler, metaHandler,
  clearTmdbCache, clearTmdbDetailsCache, clearTmdbDiscoverCache,
  clearAiCache, removeAiCacheByKeywords, purgeEmptyAiCacheEntries,
  clearRpdbCache, clearFanartCache, clearTraktCache, clearTraktRawDataCache,
  clearQueryAnalysisCache, clearSimilarContentCache,
  getCacheStats, serializeAllCaches, deserializeAllCaches,
  discoverTypeAndGenres, filterTraktDataByGenres,
  incrementQueryCounter, getQueryCount, setQueryCount,
  removeTmdbDiscoverCacheItem, listTmdbDiscoverCacheKeys,
  getRpdbTierFromApiKey, searchTMDBExactMatch, determineIntentFromKeywords,
};
