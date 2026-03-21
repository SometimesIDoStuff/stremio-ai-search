const { addonBuilder } = require("stremio-addon-sdk");
const fetch = require("node-fetch").default;
const logger = require("./utils/logger");
const path = require("path");
const { decryptConfig } = require("./utils/crypto");
const { withRetry } = require("./utils/apiRetry");
const TMDB_API_BASE = "https://api.themoviedb.org/3";
const TMDB_CACHE_DURATION = 7 * 24 * 60 * 60 * 1000;
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

// Single model for ALL queries — defaults to qwen2.5:7b (best speed/format in bench test)
const DEFAULT_LLM_BASE_URL = process.env.LLM_BASE_URL || "http://localhost:11434/v1";
const DEFAULT_LLM_MODEL = process.env.LLM_MODEL || "qwen2.5:7b";
const DEFAULT_LLM_API_KEY = process.env.LLM_API_KEY || "ollama";

let queryCounter = 0;

/**
 * OpenAI-compatible LLM call - works with Ollama, LM Studio, vLLM, etc.
 * Uses a single configured model for ALL query types.
 */
async function callLocalLLM(prompt, baseUrl, model, apiKey) {
  const url = `${baseUrl.replace(/\/$/, "")}/chat/completions`;
  const body = {
    model: model,
    messages: [{ role: "user", content: prompt }],
    temperature: 0.3,
    max_tokens: 300,
    stream: false,
  };
  const response = await withRetry(
    async () => {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(body),
      });
      if (!res.ok) { const err = new Error(`LLM API error: ${res.status}`); err.status = res.status; throw err; }
      return res.json();
    },
    { maxRetries: 1, initialDelay: 500, maxDelay: 2000, shouldRetry: (error) => !error.status || error.status >= 500, operationName: "Local LLM API call" }
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
    if (this.cache.size >= this.max) { const oldestKey = this.timestamps.keys().next().value; this.delete(oldestKey); }
    this.cache.set(key, value);
    this.timestamps.set(key, Date.now());
    if (this.ttl !== Infinity) { this.expirations.set(key, Date.now() + this.ttl); }
    return this;
  }
  get(key) {
    if (!this.cache.has(key)) return undefined;
    const expiration = this.expirations.get(key);
    if (expiration && Date.now() > expiration) { this.delete(key); return undefined; }
    this.timestamps.delete(key); this.timestamps.set(key, Date.now());
    return this.cache.get(key);
  }
  has(key) {
    if (!this.cache.has(key)) return false;
    const expiration = this.expirations.get(key);
    if (expiration && Date.now() > expiration) { this.delete(key); return false; }
    return true;
  }
  delete(key) { this.cache.delete(key); this.timestamps.delete(key); this.expirations.delete(key); return true; }
  clear() { this.cache.clear(); this.timestamps.clear(); this.expirations.clear(); return true; }
  get size() { return this.cache.size; }
  keys() { return Array.from(this.cache.keys()); }
  serialize() {
    const entries = [];
    for (const [key, value] of this.cache.entries()) { entries.push({ key, value, timestamp: this.timestamps.get(key), expiration: this.expirations.get(key) }); }
    return { max: this.max, ttl: this.ttl, entries };
  }
  deserialize(data) {
    if (!data || !data.entries) return false;
    this.max = data.max || this.max; this.ttl = data.ttl || this.ttl; this.clear();
    for (const entry of data.entries) {
      if (entry.expiration && Date.now() > entry.expiration) continue;
      this.cache.set(entry.key, entry.value); this.timestamps.set(entry.key, entry.timestamp);
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
const queryAnalysisCache = new SimpleLRUCache({ max: 1000, ttl: AI_CACHE_DURATION });

const HOST = process.env.HOST ? `https://${process.env.HOST}` : "https://stremio.itcon.au";
const PORT = 7000;
const BASE_PATH = "/aisearch";

setInterval(() => {
  logger.info("Cache statistics", {
    tmdbCache: { size: tmdbCache.size, maxSize: tmdbCache.max },
    tmdbDetailsCache: { size: tmdbDetailsCache.size, maxSize: tmdbDetailsCache.max },
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
    if (!hasMovies && !hasSeries) { aiRecommendationsCache.delete(key); purgedCount++; }
  }
  return { scanned: totalScanned, purged: purgedCount, remaining: aiRecommendationsCache.size };
}

function mergeAndDeduplicate(newItems, existingItems) {
  const existingMap = new Map();
  existingItems.forEach((item) => { const media = item.movie || item.show; const id = item.id || media?.ids?.trakt; if (id) existingMap.set(id, item); });
  newItems.forEach((item) => {
    const media = item.movie || item.show; const id = item.id || media?.ids?.trakt;
    if (id) { if (!existingMap.has(id) || (item.last_activity && existingMap.get(id).last_activity && new Date(item.last_activity) > new Date(existingMap.get(id).last_activity))) existingMap.set(id, item); }
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
  const words = message.split(' '); let lines = []; let currentLine = words[0] || '';
  for (let i = 1; i < words.length; i++) { let testLine = currentLine + ' ' + words[i]; if (testLine.length > 35) { lines.push(currentLine); currentLine = words[i]; } else currentLine = testLine; }
  lines.push(currentLine);
  const messageTspans = lines.map((line, index) => `<tspan x="250" y="${560 + index * 30}">${line}</tspan>`).join('');
  const svg = `<svg width="500" height="750" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#2d2d2d" /><path d="M250 50 L450 400 L50 400 Z" fill="#c0392b"/><path d="M250 120 L400 380 L100 380 Z" fill="#e74c3c"/><text fill="white" font-size="60" font-family="Arial, sans-serif" x="250" y="270" text-anchor="middle" font-weight="bold">!</text><text fill="white" font-size="32" font-family="Arial, sans-serif" x="250" y="500" text-anchor="middle" font-weight="bold">${title}</text><text fill="white" font-size="24" font-family="Arial, sans-serif" text-anchor="middle">${messageTspans}</text><text fill="#bdc3c7" font-size="20" font-family="Arial, sans-serif" x="250" y="700" text-anchor="middle">Please check the addon configuration.</text></svg>`;
  return { id: `error:${title.replace(/\s+/g, '_')}`, type: 'movie', name: title, description: message, poster: `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`, posterShape: 'regular' };
}

async function fetchTraktIncrementalData(clientId, accessToken, type, lastUpdate) {
  const startDate = new Date(lastUpdate).toISOString().split(".")[0] + "Z";
  const endpoints = [
    `${TRAKT_API_BASE}/users/me/watched/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
    `${TRAKT_API_BASE}/users/me/ratings/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
    `${TRAKT_API_BASE}/users/me/history/${type}?extended=full&start_at=${startDate}&page=1&limit=100`,
  ];
  const headers = { "Content-Type": "application/json", "trakt-api-version": "2", "trakt-api-key": clientId, Authorization: `Bearer ${accessToken}` };
  const responses = await Promise.all(endpoints.map((endpoint) =>
    fetch(endpoint, { headers }).then((res) => res.json()).catch((err) => { logger.error("Trakt API Error:", { endpoint, error: err.message }); return []; })
  ));
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
  if (traktCache.has(processedCacheKey)) { return traktCache.get(processedCacheKey).data; }
  let rawData; let isIncremental = false;
  if (traktRawDataCache.has(rawCacheKey)) {
    const cachedRaw = traktRawDataCache.get(rawCacheKey);
    const lastUpdate = cachedRaw.lastUpdate || cachedRaw.timestamp;
    try {
      const newData = await fetchTraktIncrementalData(clientId, accessToken, type, lastUpdate);
      rawData = { watched: mergeAndDeduplicate(newData.watched, cachedRaw.data.watched), rated: mergeAndDeduplicate(newData.rated, cachedRaw.data.rated), history: mergeAndDeduplicate(newData.history, cachedRaw.data.history), lastUpdate: Date.now() };
      isIncremental = true;
      traktRawDataCache.set(rawCacheKey, { timestamp: Date.now(), lastUpdate: Date.now(), data: rawData });
    } catch (error) { logger.error("Incremental Trakt update failed", { error: error.message }); isIncremental = false; }
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
    } catch (error) { logger.error("Trakt API Error:", { error: error.message }); return null; }
  }
  const preferences = await processPreferencesInParallel(rawData.watched, rawData.rated, rawData.history);
  const result = { watched: rawData.watched, rated: rawData.rated, history: rawData.history, preferences, lastUpdate: rawData.lastUpdate, isIncrementalUpdate: isIncremental };
  traktCache.set(processedCacheKey, { timestamp: Date.now(), data: result });
  return result;
}

async function searchTMDB(title, type, year, tmdbKey, language = "en-US", includeAdult = false) {
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
        const error = new Error(errorMessage); error.status = searchResponse.status; error.isRateLimit = searchResponse.status === 429; error.isInvalidKey = searchResponse.status === 401; throw error;
      }
      return searchResponse.json();
    }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB search", shouldRetry: (error) => !error.isInvalidKey && (!error.status || error.status >= 500 || error.isRateLimit) });
    if (responseData?.results?.[0]) {
      const result = responseData.results[0];
      const tmdbData = { poster: result.poster_path ? `https://image.tmdb.org/t/p/w500${result.poster_path}` : null, backdrop: result.backdrop_path ? `https://image.tmdb.org/t/p/original${result.backdrop_path}` : null, tmdbRating: result.vote_average, genres: result.genre_ids, overview: result.overview || "", tmdb_id: result.id, title: result.title || result.name, release_date: result.release_date || result.first_air_date };
      if (!tmdbData.imdb_id) {
        const detailsCacheKey = `details_${searchType}_${result.id}_${language}`;
        let detailsData;
        if (tmdbDetailsCache.has(detailsCacheKey)) { detailsData = tmdbDetailsCache.get(detailsCacheKey).data; }
        else {
          const detailsUrl = `${TMDB_API_BASE}/${searchType}/${result.id}?api_key=${tmdbKey}&append_to_response=external_ids&language=${language}`;
          detailsData = await withRetry(async () => { const r = await fetch(detailsUrl); if (!r.ok) throw new Error(`TMDB details error: ${r.status}`); return r.json(); }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB details" });
          tmdbDetailsCache.set(detailsCacheKey, { timestamp: Date.now(), data: detailsData });
        }
        if (detailsData) tmdbData.imdb_id = detailsData.imdb_id || detailsData.external_ids?.imdb_id;
      }
      tmdbCache.set(cacheKey, { timestamp: Date.now(), data: tmdbData });
      return tmdbData;
    }
    tmdbCache.set(cacheKey, { timestamp: Date.now(), data: null });
    return null;
  } catch (error) { logger.error("TMDB Search Error:", { error: error.message, params: { title, type, year } }); return null; }
}

async function searchTMDBExactMatch(title, type, tmdbKey, language = "en-US", includeAdult = false) {
  const cacheKey = `tmdb_search_${title}-${type}-${language}-adult:${includeAdult}`;
  if (tmdbCache.has(cacheKey)) {
    const responseData = tmdbCache.get(cacheKey).data;
    if (responseData && responseData.length > 0) { const normalizedTitle = title.toLowerCase().trim(); const exactMatch = responseData.find((r) => (r.title || r.name || "").toLowerCase().trim() === normalizedTitle); return { isExactMatch: !!exactMatch, results: responseData }; }
    return { isExactMatch: false, results: [] };
  }
  try {
    const searchType = type === "movie" ? "movie" : "tv";
    const searchParams = new URLSearchParams({ api_key: tmdbKey, query: title, include_adult: includeAdult, language: language });
    const searchUrl = `${TMDB_API_BASE}/search/${searchType}?${searchParams.toString()}`;
    const responseData = await withRetry(async () => { const r = await fetch(searchUrl); if (!r.ok) throw new Error(`TMDB search error: ${r.status}`); return r.json(); }, { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, operationName: "TMDB exact match search", shouldRetry: (e) => !e.status || e.status >= 500 });
    const results = responseData?.results || [];
    tmdbCache.set(cacheKey, { timestamp: Date.now(), data: results });
    if (results.length > 0) { const normalizedTitle = title.toLowerCase().trim(); const exactMatch = results.find((r) => (r.title || r.name || "").toLowerCase().trim() === normalizedTitle); return { isExactMatch: !!exactMatch, results }; }
    return { isExactMatch: false, results: [] };
  } catch (error) { logger.error("TMDB Search Error:", { error: error.message }); return { isExactMatch: false, results: [] }; }
}

const manifest = {
  id: "au.itcon.aisearch",
  version: "1.0.67-local-llm",
  name: "AI Search (Local LLM)",
  description: "AI-powered movie and series recommendations using your local LLM",
  resources: ["catalog", "meta", { name: "stream", types: ["movie", "series"], idPrefixes: ["tt"] }],
  types: ["movie", "series"],
  catalogs: [
    { type: "movie", id: "aisearch.top", name: "AI Movie Search", extra: [{ name: "search", isRequired: true }, { name: "skip" }], isSearch: true },
    { type: "series", id: "aisearch.top", name: "AI Series Search", extra: [{ name: "search", isRequired: true }, { name: "skip" }], isSearch: true },
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
  const seriesKeywords = { strong: [/\bseries\b/, /\btv show(s)?\b/, /\btelevision\b/, /\bshow(s)?\b/, /\bepisode(s)?\b/, /\bseason(s)?\b/, /\bdocumentary?\b/, /\bdocumentaries?\b/], medium: [/\bpilot\b/, /\bfinale\b/], weak: [/\bcharacter\b/, /\bcast\b/, /\bplot\b/, /\bstoryline\b/, /\bnarrative\b/] };
  let movieScore = 0; let seriesScore = 0;
  for (const p of movieKeywords.strong) { if (p.test(normalizedQuery)) movieScore += 3; }
  for (const p of movieKeywords.medium) { if (p.test(normalizedQuery)) movieScore += 2; }
  for (const p of movieKeywords.weak) { if (p.test(normalizedQuery)) movieScore += 1; }
  for (const p of seriesKeywords.strong) { if (p.test(normalizedQuery)) seriesScore += 3; }
  for (const p of seriesKeywords.medium) { if (p.test(normalizedQuery)) seriesScore += 2; }
  for (const p of seriesKeywords.weak) { if (p.test(normalizedQuery)) seriesScore += 1; }
  // Streaming services carry both movies and series — do not bias toward series
  if (/\b(cinema|theatrical|box office|imax)\b/.test(normalizedQuery)) movieScore += 1;
  if (/\b\d{4}-\d{4}\b/.test(normalizedQuery)) seriesScore += 1;
  const diff = Math.abs(movieScore - seriesScore);
  if (diff < 2) return "ambiguous";
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
  const excludedGenres = new Set(); let match;
  while ((match = notPattern.exec(q)) !== null) {
    const negatedTerm = match[1].toLowerCase().trim();
    if (supportedGenres.has(negatedTerm)) { excludedGenres.add(genreAliases[negatedTerm] || negatedTerm); }
    else { for (const [genre, pattern] of Object.entries(basicGenres)) { if (pattern.test(negatedTerm)) { excludedGenres.add(genre); break; } } for (const [genre, pattern] of Object.entries(subGenres)) { if (pattern.test(negatedTerm)) { excludedGenres.add(genre); break; } } }
  }
  const genres = { include: [], exclude: Array.from(excludedGenres), mood: [], style: [] };
  const combinedMatch = q.match(combinedPattern);
  if (combinedMatch) genres.include.push(combinedMatch[0].toLowerCase().replace(/\s+/g, "-"));
  for (const [genre, pattern] of Object.entries(basicGenres)) { if (pattern.test(q) && !excludedGenres.has(genre)) { const gi = q.search(pattern); const bg = q.substring(0, gi); if (!bg.match(/\b(not|no|except|excluding)\s+$/)) genres.include.push(genre); } }
  for (const [subgenre, pattern] of Object.entries(subGenres)) { if (pattern.test(q) && !excludedGenres.has(subgenre)) { const gi = q.search(pattern); const bg = q.substring(0, gi); if (!bg.match(/\b(not|no|except|excluding)\s+$/)) genres.include.push(subgenre); } }
  for (const [mood, pattern] of Object.entries(moods)) { if (pattern.test(q)) genres.mood.push(mood); }
  return Object.values(genres).some((arr) => arr.length > 0) ? genres : null;
}

function isRecommendationQuery(query) {
  const q = query.toLowerCase().trim();
  if (q.startsWith("recommend")) return true;
  // Platform+genre queries go to TMDB discover, not LLM
  const platforms = /netflix|amazon|hbo|hulu|disney+|apple tv|peacock|paramount|starz|showtime|crunchyroll|tubi|plex/;
  if (platforms.test(q)) return false;
  return false;
}

function isItemWatchedOrRated(item, watchHistory, ratedItems) {
  if (!item) return false;
  const normalizedName = item.name.toLowerCase().trim(); const itemYear = parseInt(item.year);
  const isWatched = watchHistory && watchHistory.length > 0 && watchHistory.some((h) => { const media = h.movie || h.show; if (!media) return false; return media.title.toLowerCase().trim() === normalizedName && (!itemYear || !parseInt(media.year) || itemYear === parseInt(media.year)); });
  const isRated = ratedItems && ratedItems.length > 0 && ratedItems.some((r) => { const media = r.movie || r.show; if (!media) return false; return media.title.toLowerCase().trim() === normalizedName && (!itemYear || !parseInt(media.year) || itemYear === parseInt(media.year)); });
  return isWatched || isRated;
}

async function getLandscapeThumbnail(tmdbData, imdbId, fanartApiKey, tmdbKey) {
  if (fanartApiKey && imdbId) { try { const t = await fetchFanartThumbnail(imdbId, fanartApiKey); if (t) return t; } catch (e) { logger.debug("Fanart fetch failed", { error: e.message }); } }
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
    let FanartApi; try { FanartApi = require("fanart.tv"); } catch (e) { return null; }
    const fanart = new FanartApi(effectiveFanartKey);
    const data = await withRetry(async () => await fanart.movies.get(imdbId), { maxRetries: 3, baseDelay: 1000, shouldRetry: (e) => !e.status || e.status !== 401, operationName: "Fanart.tv" });
    const thumbnail = data?.moviethumb?.filter(t => t.lang === 'en' || !t.lang || t.lang.trim() === '')?.sort((a, b) => b.likes - a.likes)[0]?.url;
    fanartCache.set(cacheKey, { timestamp: Date.now(), data: thumbnail }); return thumbnail;
  } catch (error) { fanartCache.set(cacheKey, { timestamp: Date.now(), data: null }); return null; }
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
      if (!response.ok) { const error = new Error(`RPDB error: ${response.status}`); error.status = response.status; throw error; }
      return url;
    }, { maxRetries: 2, initialDelay: 1000, maxDelay: 5000, shouldRetry: (e) => e.status !== 404 && (!e.status || e.status >= 500), operationName: "RPDB poster" });
    if (isTier0User) rpdbCache.set(cacheKey, { timestamp: Date.now(), data: posterUrl }); return posterUrl;
  } catch (error) { logger.error("RPDB Error:", { error: error.message }); return null; }
}

function getRpdbTierFromApiKey(apiKey) {
  if (!apiKey) return -1;
  try { const m = apiKey.match(/^t(\d+)-/); return m && m[1] !== undefined ? parseInt(m[1]) : -1; } catch (e) { return -1; }
}


async function fetchMDBListRatings(imdbId, mdblistApiKey) {
  if (!imdbId || !mdblistApiKey) return null;
  try {
    const url = `https://mdblist.com/api/?apikey=${mdblistApiKey}&i=${imdbId}`;
    const response = await fetch(url);
    if (!response.ok) return null;
    const data = await response.json();
    if (!data || data.error) return null;
    const ratingMap = { imdb: 'IMDb', metacritic: 'Metacritic', tomatoes: 'Rotten Tomatoes', letterboxd: 'Letterboxd', trakt: 'Trakt', tmdb: 'TMDb' };
    const ratings = (data.ratings || []).filter(r => r.value && ratingMap[r.source]).map(r => ({ source: ratingMap[r.source], value: r.value, votes: r.votes || null }));
    const streams = (data.watch_providers || data.streams || []).map(s => typeof s === 'string' ? s : s.name).filter(Boolean);
    return {
      score: data.score || null,
      ratings,
      streams,
      certification: data.certification || null,
      runtime: data.runtime || null,
      released: data.released || null,
      keywords: (data.keywords || []).slice(0, 10)
    };
  } catch (e) {
    logger.debug("MDBList fetch failed", { imdbId, error: e.message });
    return null;
  }
}


function extractNetworkFromQuery(query) {
  const q = query.toLowerCase();
  const networkMap = {
    'netflix': 'Netflix',
    'amazon prime': 'Amazon Prime Video',
    'amazon': 'Amazon Prime Video',
    'hbo': 'HBO Max',
    'hulu': 'Hulu',
    'disney+': 'Disney Plus',
    'disney plus': 'Disney Plus',
    'apple tv': 'Apple TV Plus',
    'peacock': 'Peacock',
    'paramount': 'Paramount Plus',
    'starz': 'Starz',
    'showtime': 'Showtime',
    'crunchyroll': 'Crunchyroll',
    'tubi': 'Tubi TV',
    'plex': 'Plex',
  };
  for (const [keyword, name] of Object.entries(networkMap)) {
    if (q.includes(keyword)) return name;
  }
  return null;
}
async function toStremioMeta(item, platform = "unknown", tmdbKey, rpdbKey, rpdbPosterType = "poster-default", language = "en-US", config, includeAdult = false) {
  if (!item.id || !item.name) return null;
  const type = item.type || (item.id.includes("movie") ? "movie" : "series");
  const enableRpdb = config?.EnableRpdb !== undefined ? config.EnableRpdb : false;
  const userRpdbKey = config?.RpdbApiKey;
  const isTier0User = (!!userRpdbKey && getRpdbTierFromApiKey(userRpdbKey) === 0) || (!userRpdbKey && !!DEFAULT_RPDB_KEY);
  let tmdbData;
  if (item.imdbId) logger.info("toStremioMeta short-circuit check", { name: item.name, hasImdbId: !!item.imdbId, hasTmdbDetail: !!item.tmdbDetail, poster_path: item.tmdbDetail?.poster_path ?? 'NO_DETAIL' });
  if (item.imdbId && item.tmdbDetail) {
    const d = item.tmdbDetail;
    tmdbData = {
      imdb_id: item.imdbId,
      title: d.title || d.name,
      name: d.name || d.title,
      overview: d.overview || '',
      poster: d.poster_path ? `https://image.tmdb.org/t/p/w500${d.poster_path}` : null,
      backdrop: d.backdrop_path ? `https://image.tmdb.org/t/p/original${d.backdrop_path}` : null,
      genres: (d.genres || []).map(g => g.id),
      release_date: d.release_date || d.first_air_date || ''
    };
  } else {
    tmdbData = await searchTMDB(item.name, type, item.year, tmdbKey, language, includeAdult);
  }
  if (!tmdbData || !tmdbData.imdb_id) return null;
  // If short-circuit gave no poster, re-fetch via searchTMDB as fallback
  if (!tmdbData.poster && item.imdbId) {
    const fallback = await searchTMDB(item.name, type, item.year, tmdbKey, language, includeAdult);
    if (fallback?.poster) tmdbData.poster = fallback.poster;
  }
  let poster = tmdbData.poster;
  const effectiveRpdbKey = userRpdbKey || DEFAULT_RPDB_KEY;
  if (enableRpdb && effectiveRpdbKey && tmdbData.imdb_id) {
    try { const rp = await fetchRpdbPoster(tmdbData.imdb_id, effectiveRpdbKey, rpdbPosterType, isTier0User); if (rp) poster = rp; } catch (e) { logger.debug("RPDB poster failed", { error: e.message }); }
  }
  if (!poster) return null;
  const meta = { id: tmdbData.imdb_id, type, name: tmdbData.title || tmdbData.name, description: platform === "android-tv" ? (tmdbData.overview || "").slice(0, 200) : tmdbData.overview || "", year: parseInt(item.year) || 0, poster: platform === "android-tv" && poster.includes("/w500/") ? poster.replace("/w500/", "/w342/") : poster, background: tmdbData.backdrop, posterShape: "regular" };
  if (tmdbData.genres && tmdbData.genres.length > 0) meta.genres = tmdbData.genres.map((id) => (type === "series" ? TMDB_TV_GENRES[id] : TMDB_GENRES[id])).filter(Boolean);
  const mdblistKey = config?.MdblistApiKey || process.env.MDBLIST_API_KEY;
  if (mdblistKey && tmdbData.imdb_id) {
    const mdbData = await fetchMDBListRatings(tmdbData.imdb_id, mdblistKey);
    if (mdbData) {
      const imdbRating = mdbData.ratings.find(r => r.source === 'IMDb');
      if (imdbRating) meta.imdbRating = String(imdbRating.value);
      if (mdbData.certification) meta.certification = mdbData.certification;
      if (mdbData.runtime && !meta.runtime) meta.runtime = mdbData.runtime;
      if (mdbData.streams && mdbData.streams.length > 0) meta.networks = mdbData.streams;
      const ratingStr = mdbData.ratings.slice(0, 4).map(r => r.source + ': ' + r.value).join(' | ');
      const streamStr = mdbData.streams.length > 0 ? '\n📺 Streaming: ' + mdbData.streams.slice(0, 3).join(', ') : '';
      if (ratingStr) meta.description = (meta.description || '') + '\n\n⭐ ' + ratingStr + streamStr;
    }
  }
  return meta;
}

function detectPlatform(extra = {}) {
  if (extra.headers?.["stremio-platform"]) return extra.headers["stremio-platform"];
  const ua = (extra.userAgent || extra.headers?.["stremio-user-agent"] || "").toLowerCase();
  if (ua.includes("android tv") || ua.includes("chromecast") || ua.includes("androidtv")) return "android-tv";
  if (ua.includes("android") || ua.includes("mobile") || ua.includes("phone")) return "mobile";
  if (ua.includes("windows") || ua.includes("macintosh") || ua.includes("linux")) return "desktop";
  return "unknown";
}

async function getTmdbDetailsByImdbId(imdbId, type, tmdbKey, language = "en-US") {
  const cacheKey = `details_imdb_${imdbId}_${type}_${language}`;
  if (tmdbDetailsCache.has(cacheKey)) return tmdbDetailsCache.get(cacheKey).data;
  try {
    const findUrl = `${TMDB_API_BASE}/find/${imdbId}?api_key=${tmdbKey}&language=${language}&external_source=imdb_id`;
    const response = await fetch(findUrl);
    if (!response.ok) throw new Error(`TMDB find error: ${response.status}`);
    const data = await response.json();
    const results = type === 'movie' ? data.movie_results : data.tv_results;
    if (results && results.length > 0) { tmdbDetailsCache.set(cacheKey, { timestamp: Date.now(), data: results[0] }); return results[0]; }
    return null;
  } catch (error) { logger.error("TMDB details by IMDB ID error", { imdbId, error: error.message }); return null; }
}

const TMDB_GENRES = { 28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western" };
const TMDB_TV_GENRES = { 10759: "Action & Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family", 10762: "Kids", 9648: "Mystery", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics", 37: "Western" };

function clearTmdbCache() { const s = tmdbCache.size; tmdbCache.clear(); return { cleared: true, previousSize: s }; }
function clearTmdbDetailsCache() { const s = tmdbDetailsCache.size; tmdbDetailsCache.clear(); return { cleared: true, previousSize: s }; }
function clearAiCache() { const s = aiRecommendationsCache.size; aiRecommendationsCache.clear(); return { cleared: true, previousSize: s }; }
function removeAiCacheByKeywords(keywords) {
  if (!keywords || typeof keywords !== "string") throw new Error("Invalid keywords");
  const searchPhrase = keywords.toLowerCase().trim(); const removedEntries = []; let totalRemoved = 0;
  for (const key of aiRecommendationsCache.keys()) { const query = key.split("_")[0].toLowerCase(); if (query.includes(searchPhrase)) { const entry = aiRecommendationsCache.get(key); if (entry) { removedEntries.push({ key, timestamp: new Date(entry.timestamp).toISOString() }); aiRecommendationsCache.delete(key); totalRemoved++; } } }
  return { removed: totalRemoved, entries: removedEntries };
}
function clearRpdbCache() { const s = rpdbCache.size; rpdbCache.clear(); return { cleared: true, previousSize: s }; }
function clearFanartCache() { const s = fanartCache.size; fanartCache.clear(); return { cleared: true, previousSize: s }; }
function clearTraktCache() { const s = traktCache.size; traktCache.clear(); return { cleared: true, previousSize: s }; }
function clearTraktRawDataCache() { const s = traktRawDataCache.size; traktRawDataCache.clear(); return { cleared: true, previousSize: s }; }
function clearQueryAnalysisCache() { const s = queryAnalysisCache.size; queryAnalysisCache.clear(); return { cleared: true, previousSize: s }; }
function clearSimilarContentCache() { const s = similarContentCache.size; similarContentCache.clear(); return { cleared: true, previousSize: s }; }

function getCacheStats() {
  return {
    tmdbCache: { size: tmdbCache.size, maxSize: tmdbCache.max, usagePercentage: ((tmdbCache.size / tmdbCache.max) * 100).toFixed(2) + "%" },
    tmdbDetailsCache: { size: tmdbDetailsCache.size, maxSize: tmdbDetailsCache.max, usagePercentage: ((tmdbDetailsCache.size / tmdbDetailsCache.max) * 100).toFixed(2) + "%" },
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
  return { tmdbCache: tmdbCache.serialize(), tmdbDetailsCache: tmdbDetailsCache.serialize(), aiRecommendationsCache: aiRecommendationsCache.serialize(), rpdbCache: rpdbCache.serialize(), fanartCache: fanartCache.serialize(), traktCache: traktCache.serialize(), traktRawDataCache: traktRawDataCache.serialize(), queryAnalysisCache: queryAnalysisCache.serialize(), similarContentCache: similarContentCache.serialize(), stats: { queryCounter } };
}

function deserializeAllCaches(data) {
  const results = {};
  if (data.tmdbCache) results.tmdbCache = tmdbCache.deserialize(data.tmdbCache);
  if (data.tmdbDetailsCache) results.tmdbDetailsCache = tmdbDetailsCache.deserialize(data.tmdbDetailsCache);
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

async function discoverTypeAndGenres(query, llmBaseUrl, llmModel, llmApiKey) {
  const promptText = `Analyze this recommendation query: "${query}"

Respond in a single line:
type|genre1,genre2

type = movie, series, or ambiguous
genres = comma-separated, or "all"

Examples:
movie|action,thriller
series|comedy,drama
ambiguous|all

No extra text.`;
  try {
    const text = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);
    const firstLine = text.split("\n")[0].trim(); const parts = firstLine.split("|");
    if (parts.length !== 2) return { type: "ambiguous", genres: [] };
    let type = parts[0].trim().toLowerCase();
    if (type !== "movie" && type !== "series") type = "ambiguous";
    const genres = parts[1].split(",").map((g) => g.trim()).filter((g) => g.length > 0 && g.toLowerCase() !== "ambiguous");
    if (genres.length === 1 && genres[0].toLowerCase() === "all") return { type, genres: [] };
    return { type, genres };
  } catch (error) { logger.error("Genre discovery error", { error: error.message }); return { type: "ambiguous", genres: [] }; }
}

function filterTraktDataByGenres(traktData, genres) {
  if (!traktData || !genres || genres.length === 0) return { recentlyWatched: [], highlyRated: [], lowRated: [] };
  const genreSet = new Set(genres.map((g) => g.toLowerCase()));
  const hasMatchingGenre = (item) => { const media = item.movie || item.show; if (!media || !media.genres || media.genres.length === 0) return false; return media.genres.some((g) => genreSet.has(g.toLowerCase())); };
  return { recentlyWatched: (traktData.watched || []).filter(hasMatchingGenre).slice(0, 100), highlyRated: (traktData.rated || []).filter((i) => i.rating >= 4).filter(hasMatchingGenre).slice(0, 100), lowRated: (traktData.rated || []).filter((i) => i.rating <= 2).filter(hasMatchingGenre).slice(0, 100) };
}

function incrementQueryCounter() { queryCounter++; return queryCounter; }
function getQueryCount() { return queryCounter; }
function setQueryCount(newCount) { if (typeof newCount !== "number" || newCount < 0) throw new Error("Invalid count"); queryCounter = newCount; return queryCounter; }

const catalogHandler = async function (args, req) {
  const { id, type, extra } = args;
  let isHomepageQuery = false;

  try {
    const configData = args.config;
    if (!configData || Object.keys(configData).length === 0) return { metas: [createErrorMeta('Configuration Missing', 'Please configure the addon with your API keys.')] };

    const tmdbKey = configData.TmdbApiKey;
    const llmBaseUrl = DEFAULT_LLM_BASE_URL;
    const llmModel = DEFAULT_LLM_MODEL;
    const llmApiKey = DEFAULT_LLM_API_KEY;

    if (configData.traktConnectionError) return { metas: [createErrorMeta('Trakt Connection Failed', 'Your Trakt access has expired. Please log in again via the addon configuration page.')] };
    if (!tmdbKey || tmdbKey.length < 10) return { metas: [createErrorMeta('TMDB API Key Invalid', 'Your TMDB API key is missing or invalid.')] };

    const tmdbValidationUrl = `https://api.themoviedb.org/3/configuration?api_key=${tmdbKey}`;
    const tmdbResponse = await fetch(tmdbValidationUrl);
    if (!tmdbResponse.ok) return { metas: [createErrorMeta('TMDB API Key Invalid', `Validation failed (Status: ${tmdbResponse.status}).`)] };
    if (!llmBaseUrl) return { metas: [createErrorMeta('LLM Not Configured', 'No LLM base URL set. Configure LlmBaseUrl or set LLM_BASE_URL on the server.')] };

    let searchQuery = "";
    if (typeof extra === "string" && extra.includes("search=")) {
      searchQuery = decodeURIComponent(extra.split("search=")[1]?.split("&")[0] || "");
    } else if (extra?.search) searchQuery = extra.search;
    const skipCount = parseInt((typeof extra === "string" ? new URLSearchParams(extra).get("skip") : extra?.skip) || "0") || 0;
    const pageOffset = Math.floor(skipCount / 20);

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
            const entry = catalogEntries[queryIndex]; const parts = entry.split(/:(.*)/s);
            if (parts.length > 1 && parts[1].trim()) searchQuery = parts[1].trim(); else searchQuery = entry;
          }
        }
        if (!searchQuery) return { metas: [createErrorMeta('Configuration Error', 'Could not find matching homepage query.')] };
      } else {
        return { metas: [createErrorMeta('Search Required', 'Please enter a search term to get AI recommendations.')] };
      }
    }

    const language = configData.TmdbLanguage || "en-US";
    const rpdbKey = configData.RpdbApiKey || DEFAULT_RPDB_KEY;
    const rpdbPosterType = configData.RpdbPosterType || "poster-default";
    let numResults = parseInt(configData.NumResults) || 20;
    if (numResults > 200) numResults = 200;
    const tmdbNumResults = Math.max(numResults, 20);

    // Multi-page TMDB fetch — always fetch at least 2 pages (max 3 pages = 60 results)
    const fetchTmdbPages = async (baseUrl) => {
      const pagesNeeded = Math.ceil(numResults / 20);
      const pagesToFetch = Math.min(Math.max(pagesNeeded, 2), 10);
      const isAnimeQuery = /\b(anime|manga|japanese|japan)\b/i.test(searchQuery);
      const sep = baseUrl.includes('?') ? '&' : '?';
      const animeExclusion = !isAnimeQuery && !baseUrl.includes('without_genres') ? '&without_keywords=210024&without_genres=16&without_original_language=ja' : '';
      const pageUrls = Array.from({length: pagesToFetch}, (_, i) => `${baseUrl}${sep}page=${pageOffset + i + 1}${animeExclusion}`);
      const pages = await Promise.all(pageUrls.map(u => fetch(u).then(x=>x.json()).catch(()=>({results:[]}))));
      const seen = new Set();
      const merged = [];
      for (const page of pages) {
        for (const item of (page.results || [])) {
          if (!seen.has(item.id)) { seen.add(item.id); merged.push(item); }
        }
      }
      return merged.slice(0, tmdbNumResults);
    };
    const enableAiCache = configData.EnableAiCache !== undefined ? configData.EnableAiCache : true;
    const includeAdult = configData.IncludeAdult === true;
    const platform = detectPlatform(extra);
    const isSearchRequest = (typeof extra === "string" && extra.includes("search=")) || !!extra?.search;

    const intent = determineIntentFromKeywords(searchQuery);
    if (intent !== "ambiguous" && intent !== type) return { metas: [] };

    const genreCriteria = extractGenreCriteria(searchQuery);

    // Early person detection via TMDB person search (works for ANY name)
    const _earlyQ = searchQuery.toLowerCase().trim();
    const _explicitPersonKw = ['directed by','movies by','films by','filmography',
      'starring','featuring','acted by','films with','movies with','works of'];
    const _hasExplicitKw = _explicitPersonKw.some(k => _earlyQ.includes(k));
    // Strip common suffixes to get candidate name: "ryan reynolds movies" -> "ryan reynolds"
    const _nameCandidate = searchQuery.trim()
      .replace(/\b(movies|films|series|shows|tv shows|filmography|directed by|starring|featuring|with|movies with|films with|works of|best|top|all|complete)\b/gi, '')
      .replace(/\b(action|adventure|comedy|drama|horror|thriller|fantasy|mystery|romance|sci.?fi|science fiction|animation|documentary|crime|western|musical|war|history|family|sport|anime)s?\b/gi, '')
      .trim();
    // Use TMDB person search to check if candidate resolves to a real person
    const _checkIsPerson = async (candidate) => {
      if (!candidate || candidate.split(/\s+/).length < 1) return false;
      const genreWords = /^(action|adventure|comedy|drama|horror|thriller|fantasy|mystery|romance|sci.?fi|science fiction|animation|documentary|crime|western|musical|war|history|family|sport|anime)s?$/i;
      if (genreWords.test(candidate.trim())) return false;
      try {
        const _pUrl = `https://api.themoviedb.org/3/search/person?api_key=${tmdbKey}&query=${encodeURIComponent(candidate)}&language=en-US`;
        const _pRes = await fetch(_pUrl).then(x => x.json());
        const _top = _pRes.results?.[0];
        return _top && _top.popularity > 10 && _top.known_for_department;
      } catch(e) { return false; }
    };
    const isPersonQuery = _hasExplicitKw || (_nameCandidate.length > 2 && await _checkIsPerson(_nameCandidate));
    let exactMatchMeta = null;
    let tmdbInitialResults = [];

    if (!isRecommendationQuery(searchQuery) && !isPersonQuery) {
      const matchResult = await searchTMDBExactMatch(searchQuery, type, tmdbKey, language, includeAdult);
      if (matchResult) {
        tmdbInitialResults = matchResult.results;
        if (matchResult.isExactMatch) {
          const normalizedTitle = searchQuery.toLowerCase().trim();
          const exactMatchData = matchResult.results.find(r => (r.title || r.name || "").toLowerCase().trim() === normalizedTitle);
          if (exactMatchData) {
            const details = await getTmdbDetailsByImdbId(exactMatchData.id, type, tmdbKey);
            if (details && details.imdb_id) {
              const exactMatchItem = { id: `exact_${exactMatchData.id}`, name: exactMatchData.title || exactMatchData.name, year: (exactMatchData.release_date || exactMatchData.first_air_date || 'N/A').substring(0, 4), type };
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
          if (discoveredGenres.length > 0) filteredTraktData = filterTraktDataByGenres(traktData, discoveredGenres);
          else filteredTraktData = { recentlyWatched: traktData.watched?.slice(0, 100) || [], highlyRated: (traktData.rated || []).filter((i) => i.rating >= 4).slice(0, 25), lowRated: (traktData.rated || []).filter((i) => i.rating <= 2).slice(0, 25) };
        }
      }
    }

    const _cacheQueryNorm = searchQuery.toLowerCase().trim().replace(/\s+/g, ' ');
    const _cacheQueryHash = require('crypto').createHash('sha1').update(_cacheQueryNorm).digest('hex').slice(0, 16);
    const cacheKey = `${_cacheQueryHash}_${type}_${traktData ? "trakt" : "no_trakt"}`;
    if (enableAiCache && !traktData && !isHomepageQuery && aiRecommendationsCache.has(cacheKey)) {
      const cached = aiRecommendationsCache.get(cacheKey);
      if (cached.configNumResults && numResults > cached.configNumResults) { aiRecommendationsCache.delete(cacheKey); }
      else if (!cached.data?.recommendations || (type === "movie" && !cached.data.recommendations.movies) || (type === "series" && !cached.data.recommendations.series)) { aiRecommendationsCache.delete(cacheKey); }
      else {
        logger.info("LLM cache hit", { key: cacheKey, query: searchQuery });
        const selectedRecommendations = type === "movie" ? cached.data.recommendations.movies || [] : cached.data.recommendations.series || [];
        if (selectedRecommendations.length === 0) return { metas: [createErrorMeta('No Results Found', 'The AI could not find any recommendations.')] };
        const metas = (await Promise.all(selectedRecommendations.map((item) => toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData, includeAdult)))).filter(Boolean);
        if (metas.length === 0 && !exactMatchMeta) return { metas: [createErrorMeta('Data Fetch Error', 'Could not retrieve details for recommendations.')] };
        let finalMetas = exactMatchMeta ? [exactMatchMeta, ...metas.filter((m) => m.id !== exactMatchMeta.id)] : metas;
        const _networkFilter = extractNetworkFromQuery(searchQuery);
        if (_networkFilter && finalMetas.some(m => m.networks && m.networks.length > 0)) {
          const filtered = finalMetas.filter(m => m.networks && m.networks.some(n => n.toLowerCase().includes(_networkFilter.toLowerCase())));
          if (filtered.length > 0) finalMetas.length = 0, finalMetas.push(...filtered);
        }
        if (finalMetas.length > 0 && isSearchRequest) incrementQueryCounter();
        return { metas: finalMetas };
      }
    }

    // === TMDB DIRECT ROUTING (before LLM) ===
    const tmdbRouter = await (async () => {
      if (isPersonQuery) return null;
      const q = searchQuery.toLowerCase().trim();
      logger.info('TMDB_ROUTER_DEBUG', { q });
      const tmdbType = type === 'series' ? 'tv' : type;

      // ── Dynamic TMDB collection search — runs FIRST before ALL other checks ──
      // Handles Star Wars, John Wick, Conjuring, Dark Knight, HTTYD, etc.
      try {
        const colSearchUrl = `https://api.themoviedb.org/3/search/collection?api_key=${tmdbKey}&query=${encodeURIComponent(searchQuery)}&language=${language}`;
        const colSearch = await fetch(colSearchUrl).then(x=>x.json());
        const bestCol = colSearch.results?.[0];
        if (bestCol) {
          const colDetail = await fetch(`https://api.themoviedb.org/3/collection/${bestCol.id}?api_key=${tmdbKey}&language=${language}`).then(x=>x.json());
          if (colDetail.parts?.length >= 2) {
            return colDetail.parts
              .filter(p => p.release_date)
              .sort((a, b) => a.release_date.localeCompare(b.release_date))
              .slice(0, tmdbNumResults);
          }
        }
      } catch(e) { logger.error('Collection search failed', { error: e.message, query: searchQuery }); }
      // ── End dynamic collection search ────────────────────────────────────────
      const yearMatch = searchQuery.match(/\b(19\d{2}|20\d{2})\b/);
      const currentYear = new Date().getFullYear();

      // ── Movie Genre IDs
      const movieGenreMap = {
        'action':28,'action movie':28,'action film':28,
        'adventure':12,'adventure movie':12,
        'animation':16,'animated':16,'cartoon':16,'pixar':16,'disney animated':16,
        'comedy':35,'comedies':35,'funny':35,'rom-com':35,'romantic comedy':35,'dark comedy':35,'black comedy':35,'stand-up':35,'sitcom':35,
        'crime':80,'crime thriller':80,'gangster':80,'mob':80,'mafia':80,
        'documentary':99,'docuseries':99,'nature documentary':99,'true crime documentary':99,
        'drama':18,'dramas':18,'dramatic':18,'melodrama':18,
        'family':10751,'kids':10751,'children':10751,'family friendly':10751,
        'psychological thriller':53,'psychological thrillers':53,
        'fantasy':14,'fantasy movie':14,'magical':14,'fairy tale':14,'sword and sorcery':14,
        'history':36,'historical':36,'period piece':36,'period drama':36,'costume drama':36,
        'horror':27,'horror movie':27,'scary':27,'psychological horror':27,
        'music':10402,'musical':10402,'concert':10402,
        'mystery':9648,'mysteries':9648,
        'romance':10749,'romantic':10749,'love story':10749,
        'sci-fi':878,'scifi':878,'science fiction':878,'sci fi':878,'space':878,'alien invasion':878,'cyberpunk':878,'dystopian sci-fi':878,
        'thriller':53,'suspense':53,
        'tv movie':10770,'war':10752,'military':10752,'wwii':10752,'world war':10752,
        'western':37,'cowboy':37,'wild west':37,
        'superhero':28,'comic book':28,'marvel':28,'dc comics':28,
        'biography':36,'biopic':36,'biographical':36,
        'sport':10770,'sports movie':10770
      };
      // ── TV Genre IDs
      const tvGenreMap = {
        'action':10759,'action & adventure':10759,'adventure':10759,'action series':10759,
        'animation':16,'animated':16,'cartoon':16,'animated series':16,
        'comedy':35,'comedies':35,'funny':35,'sitcom':35,'dark comedy':35,'sketch comedy':35,
        'crime':80,'crime drama':80,'crime series':80,'police procedural':80,'detective':80,
        'documentary':99,'docuseries':99,'nature documentary':99,'true crime':99,
        'drama':18,'dramas':18,'dramatic':18,'teen drama':18,'medical drama':18,'legal drama':18,
        'family':10751,'kids':10762,'children':10762,'family friendly':10751,
        'mystery':9648,'mysteries':9648,'whodunit':9648,
        'news':10763,'reality':10764,'reality tv':10764,'reality show':10764,'competition':10764,
        'sci-fi':10765,'scifi':10765,'sci fi':10765,'science fiction':10765,'space':10765,'cyberpunk':10765,
        'fantasy':10765,'magical':10765,'fairy tale':10765,
        'soap':10766,'soap opera':10766,
        'talk':10767,'talk show':10767,'late night':10767,
        'war':10768,'war & politics':10768,'military':10768,'political':10768,'political drama':10768,
        'western':37,'cowboy':37,
        // TV has no Thriller genre - use keywords (12565 psych, etc.) instead
        // TV has no Horror genre - use keywords (12339 slasher, 6152 supernatural) instead
        // NOTE: psychological thriller is keyword-only (12565) - NOT in genreMap
        'romance':10749,'romantic':10749,
        'superhero':10759,'comic book':10759,'marvel':10759,'dc':10759,
        'anime':16,'manga':16,
        'biography':18,'biopic':18,'true story':18,
        'sports':10759,'sport':10759
      };
      // Unified genre lookup (uses correct map per type)
      const genreMap = tmdbType === 'tv' ? tvGenreMap : movieGenreMap;
      const getGenreId = () => { for (const [k,v] of Object.entries(genreMap)) { if (q.includes(k)) return v; } return null; };

      // ── Production Company IDs
      const studioMap = {
        'marvel':420,'mcu':420,'marvel studios':420,'dc':9993,'dc studios':9993,
        'warner bros':174,'warner':174,'universal':33,'paramount pictures':4,
        'sony':5,'columbia':5,'disney':2,'walt disney':2,'pixar':3,
        'dreamworks':521,'lionsgate':1632,'a24':41077,'blumhouse':3172,
        'new line':12,'mgm':8411,'miramax':14,'20th century fox':25,
        '20th century studios':25,'neon':130928,'focus features':10146,
        'searchlight':127928,'annapurna':108222,'amblin':56,'legendary':923,
        'orion':10829,'tristar':559,'bbc films':3268,'working title':10163
      };
      // ── Keyword IDs (all IDs verified via TMDB keyword search API)
      const keywordMap = {
        // Psychological / Thriller sub-genres
        'psychological thriller':'12565|174089|226106','psychological':'12565|174089','mind games':226106,'unreliable narrator':174089,
        'political thriller':209817,'legal thriller':254459,'spy thriller':470,'espionage':5265,
        'conspiracy':10410,'paranoia':226106,
        // Horror sub-genres
        'slasher':12339,'found footage':163053,'supernatural':6152,'demonic':15001,
        'possession':9712,'haunted house':3358,'ghost':162846,'creature feature':13031,
        'monster':1299,'zombie':12377,'vampire':3133,'werewolf':12564,'cult':6158,
        // Crime / Action sub-genres
        'heist':10051,'hitman':2708,'assassin':782,'serial killer':10714,
        'kidnapping':1930,'revenge':9748,'detective':703,'whodunit':12570,'noir':340566,
        // Sci-Fi sub-genres
        'time travel':4379,'dystopia':4565,'post-apocalyptic':4458,'alien':9951,
        'robot':14544,'artificial intelligence':310,'survival':10349,
        // Other
        'superhero':9715,'drug':251961,
      };

      // ── Person search (directors & actors) — handled via TMDB person search
      const directorKeywords = [
        'directed by','director','films by','movies by','from director',
        'tarantino','quentin tarantino','nolan','christopher nolan','scorsese',
        'martin scorsese','spielberg','steven spielberg','kubrick','stanley kubrick',
        'lynch','david lynch','fincher','david fincher','villeneuve','denis villeneuve',
        'anderson','wes anderson','paul thomas anderson','coen brothers','cohen brothers',
        'tim burton','ridley scott','james cameron','alfred hitchcock','hitchcock',
        'stanley kubrick','sofia coppola','coppola','francis ford coppola',
        'jordan peele','peele','m night shyamalan','shyamalan','zack snyder','snyder',
        'michael bay','bay','ron howard','clint eastwood','eastwood','spike lee',
        'woody allen','roman polanski','billy wilder','john ford','frank capra',
        'terry gilliam','gilliam','darren aronofsky','aronofsky','ang lee',
        'werner herzog','herzog','jean-luc godard','godard','akira kurosawa','kurosawa',
        'ingmar bergman','bergman','pedro almodovar','almodovar','guillermo del toro','del toro'
      ];
      const actorKeywords = [
        'starring','with','featuring','acted by','films with','movies with',
        'tom hanks','hanks','tom cruise','cruise','leonardo dicaprio','dicaprio',
        'brad pitt','pitt','will smith','denzel washington','denzel','morgan freeman',
        'robert de niro','de niro','al pacino','pacino','meryl streep','streep',
        'cate blanchett','blanchett','jennifer lawrence','lawrence','scarlett johansson',
        'ryan reynolds','reynolds','dwayne johnson','the rock','vin diesel','diesel',
        'chris evans','chris hemsworth','robert downey jr','rdj','samuel l jackson',
        'mark ruffalo','jeremy renner','benedict cumberbatch','cumberbatch',
        'matt damon','damon','jake gyllenhaal','gyllenhaal','joaquin phoenix','phoenix',
        'christian bale','bale','heath ledger','ledger','cillian murphy','murphy',
        'emily blunt','blunt','anne hathaway','hathaway','natalie portman','portman',
        'jessica chastain','chastain','viola davis','davis','lupita nyongo','nyongo',
        'idris elba','elba','michael fassbender','fassbender','chiwetel ejiofor',
        'keanu reeves','reeves','adam sandler','sandler','jim carrey','carrey',
        'eddie murphy','robin williams','bill murray','murray','jack nicholson','nicholson',
        'johnny depp','depp','antonio banderas','banderas','javier bardem','bardem',
        'pedro pascal','pascal','oscar isaac','isaac','andrew garfield','garfield',
        'timothee chalamet','chalamet','florence pugh','pugh','zendaya','anya taylor-joy',
        'sydney sweeney','margot robbie','robbie','gal gadot','gadot',
        'chris pratt','pratt','paul rudd','rudd','tom holland','holland',
        'harrison ford','ford','clint eastwood','eastwood','sylvester stallone','stallone',
        'arnold schwarzenegger','schwarzenegger','bruce willis','willis',
        'nicolas cage','cage','john travolta','travolta','samuel jackson'
      ];
      // Check if query is person-based
      const _innerIsPersonQuery = directorKeywords.some(k => q.includes(k)) || actorKeywords.some(k => q.includes(k));

      // ── Live TMDB person search resolver
      // Detects any name-like pattern and resolves to TMDB person ID → filmography
      const resolvePersonFromQuery = async (queryStr) => {
        // Strip common prefixes to isolate the name
        const cleaned = queryStr
          .replace(/\b(movies?|films?|series?|shows?|tv|directed by|director|starring|with|featuring|by|from|of|best|top|greatest|all|watchlist)\b/gi, '')
          .replace(/\s+/g, ' ').trim();
        const genreWords = /^(action|adventure|comedy|drama|horror|thriller|fantasy|mystery|romance|sci.?fi|science fiction|animation|documentary|crime|western|musical|war|history|family|sport|anime)s?$/i;
        if (!cleaned || cleaned.length < 3 || genreWords.test(cleaned.trim())) return null;
        try {
          const url = `https://api.themoviedb.org/3/search/person?api_key=${tmdbKey}&query=${encodeURIComponent(cleaned)}&language=en-US`;
          const res = await fetch(url);
          const data = await res.json();
          if (!data.results?.length) return null;
          const person = data.results[0];
          if (person.popularity < 10) return null; // skip nobodies
          return { id: person.id, name: person.name, known_for_department: person.known_for_department };
        } catch(e) { return null; }
      };

      const fetchPersonFilmography = async (personId, personDept, tmdbType, origQuery) => {
        try {
          const creditsType = tmdbType === 'movie' ? 'movie_credits' : 'tv_credits';
          const url = `https://api.themoviedb.org/3/person/${personId}/${creditsType}?api_key=${tmdbKey}&language=en-US`;
          const res = await fetch(url);
          const data = await res.json();
          // Combine cast + crew, sort by popularity, dedupe
          const isDirectorQuery = /\b(directed by|director|films by|movies by)\b/i.test(origQuery) || personDept === 'Directing';
          let items = [];
          if (isDirectorQuery) {
            items = (data.crew || []).filter(x => x.job === 'Director');
          } else {
            items = [...(data.cast || []), ...(data.crew || []).filter(x => x.job === 'Director')];
          }
          const seen = new Set();
          return items
            .filter(x => { if (seen.has(x.id)) return false; seen.add(x.id); return true; })
            .filter(x => (x.popularity || 0) > 0.5)
            .sort((a, b) => (b.popularity || 0) - (a.popularity || 0));
        } catch(e) { return []; }
      };
      // ── Sort options
      const sortMap = {
        'popular':'popularity.desc','trending':'popularity.desc',
        'top rated':'vote_average.desc','best rated':'vote_average.desc',
        'highest rated':'vote_average.desc','newest':'primary_release_date.desc',
        'latest':'primary_release_date.desc','recent':'primary_release_date.desc',
        'oldest':'primary_release_date.asc','classic':'primary_release_date.asc',
        'most voted':'vote_count.desc','box office':'revenue.desc',
        'blockbuster':'revenue.desc','revenue':'revenue.desc'
      };
      // ── Original Language codes
      const originalLangMap = {
        'french':'fr','spanish':'es','german':'de','italian':'it','japanese':'ja',
        'korean':'ko','chinese':'zh','mandarin':'zh','hindi':'hi','bollywood':'hi',
        'portuguese':'pt','russian':'ru','arabic':'ar','turkish':'tr','swedish':'sv',
        'norwegian':'no','danish':'da','dutch':'nl','polish':'pl','thai':'th',
        'indonesian':'id','hebrew':'he','greek':'el','english':'en'
      };
      // ── Decade ranges
      const decadeMap = {
        '1920s':['1920-01-01','1929-12-31'],'1930s':['1930-01-01','1939-12-31'],
        '1940s':['1940-01-01','1949-12-31'],'1950s':['1950-01-01','1959-12-31'],
        '1960s':['1960-01-01','1969-12-31'],'1970s':['1970-01-01','1979-12-31'],
        '1980s':['1980-01-01','1989-12-31'],'1990s':['1990-01-01','1999-12-31'],
        '2000s':['2000-01-01','2009-12-31'],'2010s':['2010-01-01','2019-12-31'],
        '2020s':['2020-01-01','2029-12-31']
      };
      // ── Certifications
      const certMap = {
        ' g ':' G ',' pg ':' PG ',' pg-13 ':' PG-13 ',' pg13 ':' PG-13 ',
        ' r ':' R ','r-rated':'R','nc-17':'NC-17',
        'family friendly':'PG','kids safe':'G','mature':'R',
        'tv-g':'TV-G','tv-y':'TV-Y','tv-pg':'TV-PG','tv-14':'TV-14','tv-ma':'TV-MA'
      };

      // ── Master discover param builder — uses ALL available filters ──────────
      const buildDiscoverParams = (discoverType) => {
        const isTV = discoverType === 'tv';
        const gMap = isTV ? tvGenreMap : movieGenreMap;
        const params = [];

        // Genre
        const genre = Object.entries(gMap).find(([k]) => q.includes(k));
        if (genre) params.push(`with_genres=${genre[1]}`);

        // Studio / production company
        const studio = Object.entries(studioMap).find(([k]) => q.includes(k));
        if (studio) params.push(`with_companies=${studio[1]}`);

        // Keywords
        const keyword = Object.entries(keywordMap).find(([k]) => q.includes(k));
        if (keyword) params.push(`with_keywords=${keyword[1]}`);


        // Original language (only if NOT already handled by langMap below)
        const olang = Object.entries(originalLangMap).find(([k]) => q.includes(k));
        if (olang) params.push(`with_original_language=${olang[1]}`);

        // Decade
        const decade = Object.entries(decadeMap).find(([k]) => q.includes(k));
        if (decade) {
          params.push(`primary_release_date.gte=${decade[1][0]}`);
          params.push(`primary_release_date.lte=${decade[1][1]}`);
        }

        // Specific year (if no decade)
        const yr = q.match(/(19[0-9]{2}|20[0-2][0-9])/);
        if (yr && !decade) {
          const yp = isTV ? 'first_air_date_year' : 'primary_release_year';
          params.push(`${yp}=${yr[1]}`);
        }

        // Certification (US)
        const cert = Object.entries(certMap).find(([k]) => (' '+q+' ').includes(k));
        if (cert) { params.push('certification_country=US'); params.push(`certification=${cert[1].trim()}`); }

        // Quality floor — raise bar for explicit quality queries
        const isQualityQuery = /\b(best|top|greatest|highest rated|acclaimed|award|critically|masterpiece|must.?see|all.?time)\b/.test(q);
        if (isQualityQuery) {
          params.push('vote_average.gte=7.0');
        } else if (/highly rated|best of/.test(q)) {
          params.push('vote_average.gte=7.5');
        }

        // Minimum vote count — prevents obscure zero-vote junk from ranking
        if (!params.some(p => p.startsWith('vote_count.gte'))) {
          params.push(isQualityQuery ? 'vote_count.gte=300' : 'vote_count.gte=50');
        }

        // Released-only — exclude future/unannounced content (TMDB Discover+ verified pattern)
        const _today = new Date().toISOString().split('T')[0];
        if (!isTV) {
          if (!params.some(p => p.startsWith('with_release_type'))) params.push('with_release_type=1|2|3|4|5|6');
          if (!params.some(p => p.includes('primary_release_date.lte'))) params.push(`primary_release_date.lte=${_today}`);
          if (!params.some(p => p.startsWith('region='))) params.push('region=US');
        } else {
          // Only force scripted (type=4) when not explicitly a non-scripted genre
          const isNonScripted = /\b(documentary|docuseries|reality|talk show|news|game show|variety)\b/.test(q);
          if (!isNonScripted && !params.some(p => p.startsWith('with_type='))) params.push('with_type=4');
          if (!params.some(p => p.startsWith('with_status='))) params.push('with_status=0|3|4');
        }

        // Runtime
        if (/\bshort (film|movie)\b/.test(q)) params.push('with_runtime.lte=60');
        if (/\bepic\b|\bsaga\b|\blong\b/.test(q)) params.push('with_runtime.gte=140');

        // Sort
        const sortEntry = Object.entries(sortMap).find(([k]) => q.includes(k));
        let defaultSort = 'vote_count.desc';
        if (/\b(popular|trending|hot|right now|this week|this month)\b/.test(q)) defaultSort = 'popularity.desc';
        if (/\b(best|top|greatest|highest rated|acclaimed|must.?see|all.?time)\b/.test(q)) defaultSort = 'vote_average.desc';
        if (/\b(new|latest|recent|just released|newest|upcoming)\b/.test(q)) defaultSort = 'primary_release_date.desc';
        params.push(`sort_by=${sortEntry ? sortEntry[1] : defaultSort}`);
        if (defaultSort === 'vote_average.desc' && !params.some(p => p.startsWith('vote_count.gte'))) params.push('vote_count.gte=500');

        // Watch provider
        const provider = Object.entries(platformProviderMap).find(([k]) => q.includes(k));
        if (provider) { params.push(`with_watch_providers=${provider[1]}`); params.push('watch_region=US'); }

        // Network (TV originals)
        if (isTV) {
          const network = Object.entries(platformNetworkMap).find(([k]) => q.includes(k));
          if (network) params.push(`with_networks=${network[1]}`);
        }

        // Exclude anime — triple-layer filter unless explicitly requested
        if (!/\b(anime|manga|japanese|japan|korean|chinese|bollywood)\b/i.test(q)) {
          params.push("without_keywords=210024");
          params.push("without_genres=16");
          params.push("without_original_language=ja");
        }
        return params.join('&');
      };

      // ── Person/actor/director query routing
      if (isPersonQuery) {
        const person = await resolvePersonFromQuery(searchQuery);
        if (person) {
          const items = await fetchPersonFilmography(person.id, person.known_for_department, tmdbType, searchQuery);
          if (items.length) {
            logger.info('Person query resolved', { name: person.name, id: person.id, results: items.length });
            return items;
          }
        }
      }

      // Trending/popular
      if (/\b(trending|most popular|what.s popular|popular right now)\b/.test(q)) {
        const url = `https://api.themoviedb.org/3/trending/${tmdbType}/week?api_key=${tmdbKey}`;
        const r = await fetchTmdbPages(url);
        return r.length ? r : null;
      }
      // Top rated
      if (/\b(top rated|best of all time|highest rated|greatest|all time best)\b/.test(q)) {
        const genreId = getGenreId();
        const url = genreId
          ? `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&with_genres=${genreId}&sort_by=vote_average.desc&vote_count.gte=500&language=${language}&without_genres=16&without_keywords=210024&without_original_language=ja`
          : `https://api.themoviedb.org/3/${tmdbType}/top_rated?api_key=${tmdbKey}&language=${language}&without_genres=16&without_keywords=210024&without_original_language=ja`;
        const r = await fetchTmdbPages(url);
        return r.length ? r : null;
      }
      // New/recent releases
      if (/\b(new releases?|now playing|just released|latest releases?|coming soon|upcoming)\b/.test(q)) {
        let url;
        if (type === 'movie') {
          url = `https://api.themoviedb.org/3/movie/now_playing?api_key=${tmdbKey}&language=${language}&without_genres=16&without_keywords=210024&without_original_language=ja`;
        } else {
          const sinceDate = new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
          const today = new Date().toISOString().split('T')[0];
          url = `https://api.themoviedb.org/3/discover/tv?api_key=${tmdbKey}&first_air_date.gte=${sinceDate}&first_air_date.lte=${today}&sort_by=popularity.desc&language=${language}`;
        }
        const r = await fetchTmdbPages(url);
        return r.length ? r : null;
      }
      // Year-specific queries
      if (yearMatch) {
        const year = yearMatch[1];
        const genreId = getGenreId();
        const yearParam = type === 'movie' ? 'primary_release_year' : 'first_air_date_year';
        const isYearQuality = /\b(best|top|greatest|acclaimed|highest rated|must.?see)\b/.test(q);
        const yearSort = isYearQuality ? 'vote_average.desc' : 'popularity.desc';
        const yearVoteFloor = isYearQuality ? '&vote_count.gte=300&vote_average.gte=7.0' : '&vote_count.gte=50';
        const yearRelease = !isTV ? '&with_release_type=1|2|3|4|5|6&region=US&primary_release_date.lte=' + new Date().toISOString().split('T')[0] : '';
        const yearAnimeEx = !/\b(anime|manga|japanese|japan|korean|chinese|bollywood)\b/i.test(q) ? '&without_genres=16&without_keywords=210024' : '';
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&${yearParam}=${year}&sort_by=${yearSort}${yearVoteFloor}${genreId?`&with_genres=${genreId}`:''}${yearRelease}${yearAnimeEx}&language=${language}`;
        const r = await fetchTmdbPages(url);
        return r.length ? r : null;
      }
      // Decade queries e.g. '80s horror movies', 'films from the 90s'
      const decadeMatch = q.match(/\b(\d0)s\b|\b(19|20)(\d0)s?\b|\b(sixties|seventies|eighties|nineties)\b/);
      const decadeWordMap = { sixties:'1960', seventies:'1970', eighties:'1980', nineties:'1990' };
      if (decadeMatch) {
        let decadeStart;
        if (decadeMatch[4]) decadeStart = decadeWordMap[decadeMatch[4]];
        else if (decadeMatch[2]) decadeStart = decadeMatch[2] + decadeMatch[3];
        else decadeStart = (parseInt(decadeMatch[1]) < 30 ? '20' : '19') + decadeMatch[1];
        const decadeEnd = String(parseInt(decadeStart) + 9);
        const genreId = getGenreId();
        const dateGte = decadeStart + '-01-01';
        const dateLte = decadeEnd + '-12-31';
        const dateGteParam = type === 'movie' ? 'primary_release_date.gte' : 'first_air_date.gte';
        const dateLteParam = type === 'movie' ? 'primary_release_date.lte' : 'first_air_date.lte';
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&${dateGteParam}=${dateGte}&${dateLteParam}=${dateLte}&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }
      // Actor/director queries — catches ALL combination patterns using TMDB person search
      // Works with: "Ben Stiller movies", "movies with Tom Hanks", "directed by Nolan",
      // "Tarantino films", "Spielberg action", "tarantino's best", "best of Adam Sandler",
      // "funny Will Ferrell movies", "Ryan Reynolds action comedy", etc.
      const directorMatch = q.match(/\b(?:directed by|director|filmmaker)\s+([a-z]+(?:\s+[a-z]+)?)/);
      const actorMatch = q.match(/\b(?:with|starring|featuring|actor|actress)\s+([a-z]+(?:\s+[a-z]+)?)/);
      const afterOfMatch = q.match(/\b(?:best of|films of|movies of|work of|collection of)\s+([a-z]+(?:\s+[a-z]+)?)/);
      const possessiveMatch = q.match(/([a-z]+(?:\s+[a-z]+)?)'s\s+(?:best|movies?|films?|shows?|series|work|collection)/);
      // Extract ALL 2-word combos from query and test against TMDB person search
      // This catches "funny will ferrell movies" → tries "will ferrell", "funny will", etc.
      const words = q.replace(/[^a-z\s]/g, '').trim().split(/\s+/).filter(w => 
        !['movies','films','series','shows','best','good','great','top','new','old',
          'action','comedy','horror','thriller','drama','romance','funny','scary',
          'with','starring','featuring','directed','by','of','the','a','an','and',
          'in','on','from','about','like','similar','recommended','popular'].includes(w)
      );
      // Build candidate names: single words (last names), 2-word combos, 3-word combos
      const nameCandidates = new Set();
      if (directorMatch?.[1]) nameCandidates.add(directorMatch[1]);
      if (actorMatch?.[1]) nameCandidates.add(actorMatch[1]);
      if (afterOfMatch?.[1]) nameCandidates.add(afterOfMatch[1]);
      if (possessiveMatch?.[1]) nameCandidates.add(possessiveMatch[1]);
      for (let i = 0; i < words.length; i++) {
        if (words[i].length > 3) nameCandidates.add(words[i]); // single word (last name like "tarantino")
        if (words[i+1]) nameCandidates.add(words[i] + ' ' + words[i+1]); // 2-word
        if (words[i+1] && words[i+2]) nameCandidates.add(words[i] + ' ' + words[i+1] + ' ' + words[i+2]); // 3-word
      }
      const isDirectorQuery = !!directorMatch || /\b(director|directed|filmmaker|auteur)\b/.test(q);
      let personFound = null;
      for (const candidate of nameCandidates) {
        if (candidate.length < 4) continue;
        try {
          const pd = await fetch(`https://api.themoviedb.org/3/search/person?api_key=${tmdbKey}&query=${encodeURIComponent(candidate)}&language=${language}`).then(x=>x.json());
          const p = pd.results?.[0];
          const _gw = /^(action|adventure|comedy|drama|horror|thriller|fantasy|mystery|romance|sci.?fi|science fiction|animation|documentary|crime|western|musical|war|history|family|sport|anime)s?$/i;
          if (p && p.popularity > 10 && !_gw.test(candidate.trim())) { personFound = p; break; } // popularity>10 = real known person
        } catch(e) {}
      }
      const personName = personFound?.name;
      if (personName && personName.length > 3) {
        try {
          const genreId = getGenreId();
          const roleParam = isDirectorQuery ? '&with_crew=' : '&with_cast=';
          const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}${roleParam}${personFound.id}&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
          const r = await fetchTmdbPages(url);
          if (r.length) return r;
        } catch(e) {}
      }
            // Platform + genre queries e.g. 'netflix thriller', 'amazon action movies', 'hbo drama'
      // ── Watch Provider IDs (with_watch_providers)
      const platformProviderMap = {
        'netflix':8,'amazon prime':9,'amazon':9,'prime video':9,'disney+':337,
        'disney plus':337,'hulu':15,'hbo':1899,'hbo max':1899,'max':1899,
        'apple tv':350,'apple tv+':350,'peacock':386,'paramount+':531,
        'paramount plus':531,'paramount':531,'starz':43,'showtime':37,
        'crunchyroll':283,'tubi':73,'plex':538,'mubi':11,'shudder':99,
        'britbox':151,'acorn tv':87,'amc+':526,'discovery+':584,
        'fubo':257,'sling':210,'youtube premium':188,'kanopy':191,'hoopla':212
      };
      // ── Network IDs (with_networks) for TV originals
      const platformNetworkMap = {
        'netflix':213,'hbo':49,'hbo max':49,'max':49,'hulu':453,
        'disney+':2739,'disney plus':2739,'apple tv':2552,'apple tv+':2552,
        'amazon':1024,'amazon prime':1024,'prime video':1024,'peacock':3353,
        'paramount+':4330,'paramount plus':4330,'crunchyroll':1112,
        'showtime':67,'starz':318,'amc':174,'amc+':174,'fx':88,'fxx':88,
        'nbc':6,'abc':2,'cbs':16,'fox':19,'the cw':71,'cw':71,
        'bbc':4,'bbc one':4,'bbc two':9,'bbc america':1065,
        'adult swim':80,'cartoon network':56,'syfy':77,'comedy central':47,
        'mtv':33,'vh1':158,'bravo':53,'lifetime':34,'hallmark':38,
        'usa network':30,'usa':30,'tnt':41,'tbs':32,'tlc':63,
        'history':65,'a&e':129,'discovery':64,'national geographic':171,
        'nat geo':171,'travel channel':125,'food network':66,'hgtv':84,
        'espn':29,'pbs':14,'nickelodeon':13,'nick':13,'disney channel':54,
        'freeform':2354,'sky':69,'channel 4':10,'itv':8,'canal+':58,
        'arte':172,'rai':99,'netflix japan':213,'funimation':1112
      };
      const matchedPlatform = Object.entries(platformProviderMap).find(([k]) => q.includes(k));
      if (matchedPlatform) {
        // Use buildDiscoverParams for ALL filters including network/provider/genre/sort/etc
        const extraParams = buildDiscoverParams(tmdbType);
        let url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&language=${language}&${extraParams}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }
      // Pure genre + popular (no year, no mood)
      {
        const extraParams = buildDiscoverParams(tmdbType);
        logger.info('TMDB_ROUTER_DEBUG_GENRE', { extraParams, q });
        if (extraParams.includes('with_genres=') || extraParams.includes('with_keywords=') ||
            extraParams.includes('with_companies=') || extraParams.includes('with_networks=') ||
            extraParams.includes('with_watch_providers=') || extraParams.includes('with_original_language=')) {
          const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&language=${language}&${extraParams}`;
          const r = await fetchTmdbPages(url);
          if (r.length) return r;
        }
      }
      // Language/country queries e.g. 'French movies', 'Korean thrillers', 'Japanese anime'
      const langMap = {
        french:'fr', spanish:'es', german:'de', italian:'it', japanese:'ja', korean:'ko',
        chinese:'zh', portuguese:'pt', russian:'ru', hindi:'hi', arabic:'ar', swedish:'sv',
        danish:'da', norwegian:'no', dutch:'nl', turkish:'tr', polish:'pl', thai:'th',
        'latin american':'es', 'british':'en-GB'
      };
      const countryMap = {
        french:'FR', spanish:'ES', german:'DE', italian:'IT', japanese:'JP', korean:'KR',
        chinese:'CN', portuguese:'PT', russian:'RU', hindi:'IN', arabic:'SA', swedish:'SE',
        danish:'DK', norwegian:'NO', dutch:'NL', turkish:'TR', polish:'PL', thai:'TH'
      };
      for (const [lang, code] of Object.entries(langMap)) {
        if (q.includes(lang)) {
          const genreId = getGenreId();
          const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&with_original_language=${code}&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
          const r = await fetchTmdbPages(url);
          if (r.length) return r;
          break;
        }
      }

      // Holiday/theme queries e.g. 'Christmas movies', 'Halloween films'
      const holidayKeywords = {
        christmas: 207317, halloween: 207350, thanksgiving: 207350,
        'new year': 158431, easter: 207322, valentine: 9673,
        superhero: 9715, zombie: 12377, vampire: 10063, witch: 11622,
        'time travel': 10535, heist: 9882, survival: 10051, revenge: 10178,
        spy: 10702, assassin: 10740, dystopia: 10761, apocalypse: 10840
      };
      for (const [theme, keywordId] of Object.entries(holidayKeywords)) {
        if (q.includes(theme)) {
          const genreId = getGenreId();
          const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&with_keywords=${keywordId}&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
          const r = await fetchTmdbPages(url);
          if (r.length) return r;
          break;
        }
      }

      // Award/quality queries e.g. 'Oscar winners', 'award winning', 'critically acclaimed'
      if (/\b(oscar|academy award|golden globe|bafta|award.winning|critically acclaimed|prize.winning)\b/.test(q)) {
        const genreId = getGenreId();
        const voteMin = tmdbType === 'tv' ? 200 : 1000;
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&sort_by=vote_average.desc&vote_count.gte=${voteMin}${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }

      // Hidden gems / underrated e.g. 'hidden gems', 'underrated movies', 'overlooked films'
      if (/\b(hidden gem|underrated|overlooked|cult classic|under.the.radar|lesser.known)\b/.test(q)) {
        const genreId = getGenreId();
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&sort_by=vote_average.desc&vote_count.gte=200&vote_count.lte=3000${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }

      // Runtime queries e.g. 'short movies', 'quick films', 'long epic movies'
      if (/\b(short film|short movie|quick watch|under 90|under 2 hour)\b/.test(q)) {
        const genreId = getGenreId();
        const url = `https://api.themoviedb.org/3/discover/movie?api_key=${tmdbKey}&with_runtime.lte=90&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }
      if (/\b(long movie|long film|epic film|epic movie|over 3 hour|over 2.5 hour)\b/.test(q)) {
        const genreId = getGenreId();
        const url = `https://api.themoviedb.org/3/discover/movie?api_key=${tmdbKey}&with_runtime.gte=150&sort_by=popularity.desc${genreId ? `&with_genres=${genreId}` : ''}&language=${language}`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }

      // Kids/family safe e.g. 'kids movies', 'family friendly', 'children films', 'Disney'
      if (/\b(kids?|children|family.friendly|disney|pixar|dreamworks|animated family)\b/.test(q)) {
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&with_genres=10751&certification_country=US&certification.lte=PG&sort_by=popularity.desc&language=${language}&without_genres=16&without_keywords=210024&without_original_language=ja`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }

      // Recent by genre e.g. 'new action movies', 'latest horror', 'recent sci-fi'
      if (/\b(new|latest|recent|just out|this year|past year)\b/.test(q) && getGenreId()) {
        const genreId = getGenreId();
        const sinceDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        const today = new Date().toISOString().split('T')[0];
        const dateGteParam = type === 'movie' ? 'primary_release_date.gte' : 'first_air_date.gte';
        const dateLteParam = type === 'movie' ? 'primary_release_date.lte' : 'first_air_date.lte';
        const url = `https://api.themoviedb.org/3/discover/${tmdbType}?api_key=${tmdbKey}&${dateGteParam}=${sinceDate}&${dateLteParam}=${today}&with_genres=${genreId}&sort_by=popularity.desc&language=${language}&without_genres=16&without_keywords=210024&without_original_language=ja`;
        const r = await fetchTmdbPages(url);
        if (r.length) return r;
      }

      // Similar to X e.g. 'movies like Inception', 'similar to The Office'
      const similarMatch = q.match(/\b(?:like|similar to|fans of|if you liked?)\s+(?:the\s+)?([a-z][a-z\s]{2,30}?)(?:\s*$|\s+(?:movie|film|show|series))/);
      if (similarMatch) {
        try {
          const titleQuery = similarMatch[1].trim();
          const searchUrl = `https://api.themoviedb.org/3/search/${tmdbType}?api_key=${tmdbKey}&query=${encodeURIComponent(titleQuery)}&language=${language}`;
          const searchData = await fetch(searchUrl).then(x=>x.json());
          const sourceTitle = searchData.results?.[0];
          if (sourceTitle) {
            const url = `https://api.themoviedb.org/3/${tmdbType}/${sourceTitle.id}/recommendations?api_key=${tmdbKey}&language=${language}`;
            const r = await fetchTmdbPages(url);
            if (r.length) return r;
          }
        } catch(e) { /* fall through */ }
      }

      // Franchise/collection e.g. 'Marvel movies', 'Fast and Furious', 'James Bond films'
      const franchiseMap = {
        'marvel': 9485, 'mcu': 9485, 'avengers': 9485,
        'dc comics': 10112, 'batman': 10112, 'superman': 10112,
        'star wars': 10, 'james bond': 645, 'bond': 645,
        'fast and furious': 9485, 'jurassic': 328, 'jurassic park': 328,
        'harry potter': 1241, 'wizarding world': 1241,
        'lord of the rings': 119, 'tolkien': 119,
        'mission impossible': 87359, 'indiana jones': 84, 'john wick': 404609,
        'matrix': 2344, 'terminator': 528, 'alien': 8091, 'predator': 9362,
        'planet of the apes': 173710, 'transformers': 8650,
        'x-men': 748, 'spider-man': 556, 'iron man': 131292,
        'captain america': 131295, 'thor': 131296, 'guardians': 284433,
        'pixar': 10194, 'toy story': 10194, 'disney': 2, 'dreamworks': 521
      };
      for (const [franchise, collectionId] of Object.entries(franchiseMap)) {
        if (q.includes(franchise)) {
          try {
            const url = `https://api.themoviedb.org/3/collection/${collectionId}?api_key=${tmdbKey}&language=${language}`;
            const r = await fetch(url).then(x=>x.json());
            const parts = r.parts?.sort((a,b) => new Date(a.release_date||0) - new Date(b.release_date||0));
            if (parts?.length) return parts.slice(0, numResults);
          } catch(e) {
            // fallback to keyword search
            const url = `https://api.themoviedb.org/3/search/collection?api_key=${tmdbKey}&query=${encodeURIComponent(franchise)}&language=${language}`;
            const r = await fetch(url).then(x=>x.json());
            if (r.results?.[0]) {
              const col = await fetch(`https://api.themoviedb.org/3/collection/${r.results[0].id}?api_key=${tmdbKey}`).then(x=>x.json());
              if (col.parts?.length) return col.parts.slice(0, numResults);
            }
          }
          break;
        }
      }

      return null; // fall through to LLM
    })();
    // === PERSON QUERY ROUTING (outside tmdbRouter IIFE) ===
    if (isPersonQuery) {
      const _tmdbType = type === 'series' ? 'tv' : type;
      const _q = searchQuery.toLowerCase().trim();
      const resolvePersonFromQuery = async (queryStr) => {
        const cleaned = queryStr.replace(/\b(movies|films|series|shows|tv shows|filmography|directed by|starring|featuring|with|movies with|films with|works of|best|top|all|complete)\b/gi, '').trim();
        const genreWords = /^(action|adventure|comedy|drama|horror|thriller|fantasy|mystery|romance|sci.?fi|science fiction|animation|documentary|crime|western|musical|war|history|family|sport|anime)s?$/i;
        if (!cleaned || cleaned.length < 3 || genreWords.test(cleaned.trim())) return null;
        try {
          const url = `https://api.themoviedb.org/3/search/person?api_key=${tmdbKey}&query=${encodeURIComponent(cleaned)}&language=en-US`;
          const res = await fetch(url).then(x => x.json());
          const person = res.results?.[0];
          if (!person || person.popularity < 10) return null;
          return person;
        } catch(e) { return null; }
      };
      const fetchPersonFilmography = async (personId, dept, mediaType, origQuery) => {
        const credit = mediaType === 'tv' ? 'tv_credits' : 'movie_credits';
        try {
          const url = `https://api.themoviedb.org/3/person/${personId}/${credit}?api_key=${tmdbKey}&language=en-US`;
          const res = await fetch(url).then(x => x.json());
          const isCast = dept !== 'Directing';
          const items = isCast ? (res.cast || []) : (res.crew || []).filter(c => c.job === 'Director');
          return items
            .filter(i => i.popularity > 0.5)
            .sort((a, b) => b.popularity - a.popularity)
            .slice(0, tmdbNumResults);
        } catch(e) { return []; }
      };
      const _person = await resolvePersonFromQuery(searchQuery);
      if (_person) {
        const _items = await fetchPersonFilmography(_person.id, _person.known_for_department, _tmdbType, searchQuery);
        if (_items.length > 0) {
          logger.info('Person query resolved (outer)', { name: _person.name, id: _person.id, results: _items.length });
          const metas = (await Promise.all(_items.map(async r => {
            try {
              const detailUrl = `https://api.themoviedb.org/3/${_tmdbType === 'tv' ? 'tv' : 'movie'}/${r.id}?api_key=${tmdbKey}&language=${language}&append_to_response=external_ids`;
              const detail = await fetch(detailUrl).then(x => x.json());
              const imdbId = detail.imdb_id || detail.external_ids?.imdb_id;
              const item = {
                id: imdbId ? `tt${imdbId.replace('tt','')}` : `tmdb_direct_${r.id}`,
                name: r.title || r.name,
                year: parseInt((r.release_date || r.first_air_date || '0').substring(0,4)),
                type,
                tmdbId: r.id
              };
              return toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData, includeAdult);
            } catch(e) { return null; }
          }))).filter(Boolean);
          if (metas.length > 0) return { metas };
        }
      }
      return { metas: [createErrorMeta('No Results', `Could not find filmography for "${searchQuery}"`)] };
    }
    // === END PERSON QUERY ROUTING ===

    if (tmdbRouter && tmdbRouter.length > 0) {
      logger.info("TMDB direct routing", { query: searchQuery, results: tmdbRouter.length });
      const items = tmdbRouter.map(r => ({
        name: r.title || r.name,
        year: parseInt((r.release_date || r.first_air_date || '0').substring(0, 4)),
        type,
        id: `tmdb_direct_${r.id}`,
        tmdbId: r.id
      }));
      const metas = (await Promise.all(items.map(async item => {
        if (item.tmdbId) {
          try {
            const detailUrl = `https://api.themoviedb.org/3/${type === 'series' ? 'tv' : 'movie'}/${item.tmdbId}?api_key=${tmdbKey}&language=${language}&append_to_response=external_ids`;
            const detail = await fetch(detailUrl).then(x => x.json());
            const imdbId = detail.imdb_id || detail.external_ids?.imdb_id;
            if (imdbId) {
              item.imdbId = imdbId;
              item.tmdbDetail = detail;
            }
          } catch(e) {}
        }
        return toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData, includeAdult);
      }))).filter(Boolean);
      if (metas.length > 0) {
        if (enableAiCache && !traktData) {
          const tmdbRouterCacheData = { recommendations: type === "movie" ? { movies: items } : { series: items } };
          aiRecommendationsCache.set(cacheKey, { timestamp: Date.now(), data: tmdbRouterCacheData, configNumResults: numResults });
          logger.info("TMDB router cached", { key: cacheKey, query: searchQuery, count: metas.length });
        }
        if (isSearchRequest) incrementQueryCounter();
        return { metas: exactMatchMeta ? [exactMatchMeta, ...metas.filter(m => m.id !== exactMatchMeta?.id)] : metas };
      }
    }
    // === END TMDB DIRECT ROUTING ===
    try {
      const currentYear = new Date().getFullYear();
      let franchiseInstruction = `your TOP PRIORITY is to list ALL official mainline movies from that franchise, followed by spin-offs.`;
      if (type === 'series') franchiseInstruction = `your TOP PRIORITY is to provide ALL television content in that franchise including series, documentaries, and specials.`;

      const promptText = `List ${numResults} ${type === 'movie' ? 'movies' : 'TV series'} for this request: "${searchQuery}". If the request mentions a specific year or time period, only include titles released in that year or period.
${type === 'movie' ? 'movie' : 'series'}|Title|Year

Example:
${type === 'movie' ? 'movie|The Dark Knight|2008' : 'series|Breaking Bad|2008'}

Start immediately with ${type === 'movie' ? 'movie' : 'series'}|`;
      logger.info("Calling local LLM", { model: llmModel, baseUrl: llmBaseUrl, query: searchQuery, type });

      const text = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);
      const lines = text.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("type|"));
      const recommendations = { movies: type === "movie" ? [] : undefined, series: type === "series" ? [] : undefined };

      for (const line of lines) {
        try {
          const parts = line.split("|"); let lineType, name, year;
          if (parts.length === 3) { [lineType, name, year] = parts.map((s) => s.trim()); }
          else if (parts.length === 2) {
            lineType = parts[0].trim(); const nameWithYear = parts[1].trim();
            const yearMatch = nameWithYear.match(/\((\d{4})\)$/);
            if (yearMatch) { year = yearMatch[1]; name = nameWithYear.substring(0, nameWithYear.lastIndexOf("(")).trim(); }
            else { const anyYearMatch = nameWithYear.match(/\b(19\d{2}|20\d{2})\b/); if (anyYearMatch) { year = anyYearMatch[0]; name = nameWithYear.replace(anyYearMatch[0], "").trim(); } else continue; }
          } else continue;
          const yearNum = parseInt(year);
          if (!lineType || !name || isNaN(yearNum)) continue;
          if (lineType === type && name && yearNum) {
            const item = { name, year: yearNum, type, id: `ai_${type}_${name.toLowerCase().replace(/[^a-z0-9]+/g, "_")}` };
            if (type === "movie") recommendations.movies.push(item);
            else if (type === "series") recommendations.series.push(item);
          }
        } catch (e) { /* skip malformed line */ }
      }

      const finalResult = { recommendations, fromCache: false };

      if (traktData && isRecommendation) {
        const watchHistory = traktData.watched.concat(traktData.history || []);
        if (finalResult.recommendations.movies) finalResult.recommendations.movies = finalResult.recommendations.movies.filter((m) => !isItemWatchedOrRated(m, watchHistory, traktData.rated));
        if (finalResult.recommendations.series) finalResult.recommendations.series = finalResult.recommendations.series.filter((s) => !isItemWatchedOrRated(s, watchHistory, traktData.rated));
      }

      const hasMovies = finalResult.recommendations.movies && finalResult.recommendations.movies.length > 0;
      const hasSeries = finalResult.recommendations.series && finalResult.recommendations.series.length > 0;
      if ((hasMovies || hasSeries) && !traktData && !isHomepageQuery && enableAiCache) {
        aiRecommendationsCache.set(cacheKey, { timestamp: Date.now(), data: finalResult, configNumResults: numResults });
      }

      const selectedRecommendations = type === "movie" ? finalResult.recommendations.movies || [] : finalResult.recommendations.series || [];
      const metas = (await Promise.all(selectedRecommendations.map((item) => toStremioMeta(item, platform, tmdbKey, rpdbKey, rpdbPosterType, language, configData)))).filter(Boolean);
      let finalMetas = exactMatchMeta ? [exactMatchMeta, ...metas.filter((m) => m.id !== exactMatchMeta.id)] : metas;
      if (finalMetas.length === 0) return { metas: [createErrorMeta('No Results Found', 'The AI could not find recommendations. Please try rephrasing.')] };
      if (finalMetas.length > 0 && isSearchRequest) incrementQueryCounter();
      // Chronological sort for franchise/universe/collection queries
      const chronoTriggers = /\b(universe|saga|franchise|collection|chronolog|harry potter|fast and furious|fast &|marvel|mcu|star wars|lord of the rings|hobbit|jurassic|transformers|mission impossible|james bond|007|john wick|purge|conjuring|saw|alien|terminator|indiana jones|pirates of the caribbean|toy story|ice age|shrek|despicable|minions|kung fu panda|incredibles|finding nemo|monsters inc|avengers|spider.?man|batman|x.men|deadpool|guardians|thor|iron man|captain america|hunger games|twilight|rocky|creed|rambo|die hard|rush hour|men in black|dragon ball|naruto|one piece|attack on titan|demon slayer)\b/i;
      if (chronoTriggers.test(searchQuery) && finalMetas.length > 1) {
        finalMetas.sort((a, b) => {
          const ya = parseInt(a.releaseInfo) || parseInt(a.year) || 9999;
          const yb = parseInt(b.releaseInfo) || parseInt(b.year) || 9999;
          return ya - yb;
        });
      }
      return { metas: finalMetas };

    } catch (error) {
      logger.error("LLM Error:", { error: error.message, query: searchQuery });
      let errorMessage = 'The local LLM failed to respond. Check that your LLM server is running.';
      if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) errorMessage = `Could not connect to LLM at ${args.config?.LlmBaseUrl || DEFAULT_LLM_BASE_URL}. Is it running?`;
      return { metas: [createErrorMeta('LLM Error', errorMessage)] };
    }
  } catch (error) {
    logger.error("Catalog error", { error: error.message });
    return { metas: [createErrorMeta('Addon Error', 'A critical error occurred. Check server logs.')] };
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
    } catch (error) { logger.error("streamHandler config error", { error: error.message }); }
  }
  const isWeb = req.headers["origin"]?.includes("web.stremio.com");
  const stremioUrlPrefix = isWeb ? "https://web.stremio.com/#" : "stremio://";
  return Promise.resolve({ streams: [{ name: "✨ AI Search", description: "Similar movies and shows.", externalUrl: `${stremioUrlPrefix}/detail/${args.type}/ai-recs:${args.id}`, behaviorHints: { notWebReady: true } }] });
};

async function fetchTmdbRelatedRecs(tmdbId, mediaType, tmdbKey) {
  if (!tmdbId || !tmdbKey) return [];
  const type = mediaType === "movie" ? "movie" : "tv";
  try {
    const recUrl = `${TMDB_API_BASE}/${type}/${tmdbId}/recommendations?language=en-US&page=1&api_key=${tmdbKey}`;
    const recData = await withRetry(
      async () => {
        const r = await fetch(recUrl);
        if (!r.ok) throw Object.assign(new Error(`TMDB rec fetch ${r.status}`), { status: r.status });
        return r.json();
      },
      { maxRetries: 3, initialDelay: 1000, maxDelay: 8000, shouldRetry: (e) => !e.status || e.status >= 500, operationName: "TMDB related recs" }
    );
    const recResults = recData?.results || [];
    let collectionResults = [];
    if (mediaType === "movie") {
      try {
        const detailUrl = `${TMDB_API_BASE}/movie/${tmdbId}?api_key=${tmdbKey}`;
        const detailData = await withRetry(
          async () => { const r = await fetch(detailUrl); if (!r.ok) throw new Error(`TMDB detail ${r.status}`); return r.json(); },
          { maxRetries: 2, initialDelay: 500, maxDelay: 4000, operationName: "TMDB movie detail for collection" }
        );
        if (detailData?.belongs_to_collection?.id) {
          const colUrl = `${TMDB_API_BASE}/collection/${detailData.belongs_to_collection.id}?api_key=${tmdbKey}`;
          const colData = await withRetry(
            async () => { const r = await fetch(colUrl); if (!r.ok) throw new Error(`TMDB collection ${r.status}`); return r.json(); },
            { maxRetries: 2, initialDelay: 500, maxDelay: 4000, operationName: "TMDB collection fetch" }
          );
          collectionResults = (colData?.parts || []).filter(p => p.id !== parseInt(tmdbId));
        }
      } catch (e) {
        logger.debug("TMDB collection fetch skipped", { tmdbId, error: e.message });
      }
    }
    return [...collectionResults, ...recResults].map((item, index) => ({ tmdbId: item.id, ranking: index + 1 }));
  } catch (error) {
    logger.error("fetchTmdbRelatedRecs error", { tmdbId, error: error.message });
    return [];
  }
}

async function fetchTraktRelatedRecs(imdbId, mediaType) {
  if (!imdbId || !DEFAULT_TRAKT_CLIENT_ID) return [];
  const type = mediaType === "movie" ? "movies" : "shows";
  try {
    const url = `${TRAKT_API_BASE}/${type}/${imdbId}/related`;
    const res = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": DEFAULT_TRAKT_CLIENT_ID,
      },
    });
    if (!res.ok) { logger.warn("fetchTraktRelatedRecs non-OK", { status: res.status, imdbId }); return []; }
    const json = await res.json();
    return (Array.isArray(json) ? json : [])
      .map((item, index) => ({ imdbId: item.ids?.imdb, tmdbId: item.ids?.tmdb, ranking: index + 1 }))
      .filter(r => r.imdbId);
  } catch (error) {
    logger.error("fetchTraktRelatedRecs error", { imdbId, error: error.message });
    return [];
  }
}

function mergeAndRankRecs(...recArrays) {
  const combined = recArrays.flat().filter(Boolean);
  const recMap = {};
  for (const media of combined) {
    if (!media.imdbId) continue;
    if (!recMap[media.imdbId]) {
      recMap[media.imdbId] = { ...media, count: 1, rankingSum: media.ranking };
    } else {
      recMap[media.imdbId].count += 1;
      recMap[media.imdbId].rankingSum += media.ranking;
    }
  }
  return Object.values(recMap).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return (a.rankingSum / a.count) - (b.rankingSum / b.count);
  });
}

const metaHandler = async function (args) {
  const { type, id, config } = args;
  try {
    if (!id || !id.startsWith('ai-recs:')) return { meta: null };
    if (config) {
      const decryptedConfigStr = decryptConfig(config);
      if (!decryptedConfigStr) throw new Error("Failed to decrypt config");
      const configData = JSON.parse(decryptedConfigStr);
      const { TmdbApiKey, NumResults, RpdbApiKey, RpdbPosterType, TmdbLanguage, FanartApiKey } = configData;
      const llmBaseUrl = DEFAULT_LLM_BASE_URL;
      const llmModel = DEFAULT_LLM_MODEL;
      const llmApiKey = DEFAULT_LLM_API_KEY;
      const originalId = id.split(':')[1];
      const cacheKey = `similar_${originalId}_${type}_${NumResults || 15}`;
      const cached = similarContentCache.get(cacheKey);
      if (cached) return { meta: cached.data };
      let sourceDetails = await getTmdbDetailsByImdbId(originalId, type, TmdbApiKey);
      if (!sourceDetails) { const fallbackType = type === 'movie' ? 'series' : 'movie'; sourceDetails = await getTmdbDetailsByImdbId(originalId, fallbackType, TmdbApiKey); }
      if (!sourceDetails) throw new Error(`Could not find source for: ${originalId}`);
      const sourceTitle = sourceDetails.title || sourceDetails.name;
      const sourceYear = (sourceDetails.release_date || sourceDetails.first_air_date || "").substring(0, 4);
      const sourceTmdbId = sourceDetails.id;
      let numResults = parseInt(NumResults) || 8;
      if (numResults > 25) numResults = 25;
      const [tmdbRecs, traktRecs] = await Promise.all([
        fetchTmdbRelatedRecs(sourceTmdbId, type, TmdbApiKey).catch(() => []),
        fetchTraktRelatedRecs(originalId, type).catch(() => []),
      ]);
      logger.info("Raw rec counts before merge", { tmdbCount: tmdbRecs.length, traktCount: traktRecs.length, sourceTitle });
      const resolvedTmdbRecs = await Promise.all(
        tmdbRecs.map(async (rec) => {
          if (rec.imdbId) return rec;
          try {
            const extUrl = `${TMDB_API_BASE}/${type === 'movie' ? 'movie' : 'tv'}/${rec.tmdbId}/external_ids?api_key=${TmdbApiKey}`;
            const extRes = await fetch(extUrl);
            const extJson = await extRes.json();
            return extJson?.imdb_id ? { ...rec, imdbId: extJson.imdb_id } : null;
          } catch { return null; }
        })
      );
      const ranked = mergeAndRankRecs(resolvedTmdbRecs.filter(Boolean), traktRecs);
      logger.info("Merged and ranked recs", { sourceTitle, totalRanked: ranked.length });
      let videos = [];
      if (ranked.length > 0) {
        const topRecs = ranked.slice(0, numResults);
        const videoPromises = topRecs.map(async (rec) => {
          try {
            const tmdbData = await getTmdbDetailsByImdbId(rec.imdbId, type, TmdbApiKey);
            if (!tmdbData) return null;
            let description = tmdbData.overview || "";
            if (tmdbData.vote_average && tmdbData.vote_average > 0) description = `⭐ TMDB: ${tmdbData.vote_average.toFixed(1)}/10\n\n${description}`;
            if (rec.count > 1) description = `🔗 ${rec.count} sources agree\n${description}`;
            const landscapeThumbnail = await getLandscapeThumbnail(tmdbData, rec.imdbId, FanartApiKey, TmdbApiKey);
            return { id: rec.imdbId, title: tmdbData.title || tmdbData.name, released: new Date(tmdbData.release_date || tmdbData.first_air_date || '1970-01-01').toISOString(), overview: description, thumbnail: landscapeThumbnail };
          } catch (e) { logger.debug("Video entry build failed", { imdbId: rec.imdbId, error: e.message }); return null; }
        });
        videos = (await Promise.all(videoPromises)).filter(Boolean);
      }
      if (videos.length === 0) {
        logger.info("TMDB+Trakt returned no results, falling back to LLM", { sourceTitle });
        const promptText = `You are an expert recommendation engine.\nGenerate exactly ${numResults} recommendations similar to "${sourceTitle} (${sourceYear})".\n\nPART 1: All other official ${type === 'movie' ? 'movies' : 'series'} from the same franchise as "${sourceTitle}", chronologically.\nPART 2: Fill remaining slots with highly similar unrelated titles by relevance.\n\nDo NOT include "${sourceTitle} (${sourceYear})" itself.\nNo headers or explanations.\n\nFormat:\ntype|name|year\n\nExample:\nmovie|Batman Begins|2005`;
        const responseText = await callLocalLLM(promptText, llmBaseUrl, llmModel, llmApiKey);
        const lines = responseText.split('\n').map(l => l.trim()).filter(Boolean);
        const llmVideoPromises = lines.map(async (line) => {
          const parts = line.split('|'); if (parts.length !== 3) return null;
          const [recType, name, year] = parts.map(p => p.trim());
          const tmdbData = await searchTMDB(name, recType, year, TmdbApiKey);
          if (tmdbData && tmdbData.imdb_id) {
            let description = tmdbData.overview || "";
            if (tmdbData.tmdbRating && tmdbData.tmdbRating > 0) description = `⭐ TMDB: ${tmdbData.tmdbRating.toFixed(1)}/10\n\n${description}`;
            const landscapeThumbnail = await getLandscapeThumbnail(tmdbData, tmdbData.imdb_id, FanartApiKey, TmdbApiKey);
            return { id: tmdbData.imdb_id, title: tmdbData.title, released: new Date(tmdbData.release_date || '1970-01-01').toISOString(), overview: description, thumbnail: landscapeThumbnail };
          }
          return null;
        });
        videos = (await Promise.all(llmVideoPromises)).filter(Boolean);
      }
      const meta = { id, type: 'series', name: `AI: Similar to ${sourceTitle}`, description: `Titles similar to ${sourceTitle} (${sourceYear}).`, poster: sourceDetails.poster_path ? `https://image.tmdb.org/t/p/w500${sourceDetails.poster_path}` : null, background: sourceDetails.backdrop_path ? `https://image.tmdb.org/t/p/original${sourceDetails.backdrop_path}` : null, videos };
      if (videos.length > 0) similarContentCache.set(cacheKey, { timestamp: Date.now(), data: meta });
      return { meta };
    }
  } catch (error) { logger.error("Meta Handler Error:", { message: error.message, id }); }
  return { meta: null };
};

builder.defineCatalogHandler(catalogHandler);
builder.defineStreamHandler(streamHandler);
builder.defineMetaHandler(metaHandler);

const addonInterface = builder.getInterface();

module.exports = {
  builder, addonInterface, catalogHandler, streamHandler, metaHandler,
  clearTmdbCache, clearTmdbDetailsCache,
  clearAiCache, removeAiCacheByKeywords, purgeEmptyAiCacheEntries,
  clearRpdbCache, clearFanartCache, clearTraktCache, clearTraktRawDataCache,
  clearQueryAnalysisCache, clearSimilarContentCache,
  getCacheStats, serializeAllCaches, deserializeAllCaches,
  discoverTypeAndGenres, filterTraktDataByGenres,
  incrementQueryCounter, getQueryCount, setQueryCount,
  getRpdbTierFromApiKey, searchTMDBExactMatch, determineIntentFromKeywords,
};
