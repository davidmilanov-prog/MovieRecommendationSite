import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // holds the dropdown options returned from the backend
  const [personas, setPersonas] = useState([]);

  // the currently selected dropdown value
  const [selectedPersonaId, setSelectedPersonaId] = useState("");

  // the user's natural language search input
  const [query, setQuery] = useState("");

  // list of movies returned from the backend
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // when you click "Random User", we add a temporary dropdown option for that real user
  const [tempUser, setTempUser] = useState(null);

  useEffect(() => {
    // fetch persona/archetype dropdown options once when the app loads
    axios.get('http://localhost:8000/personas')
      .then(res => {
        // convert ids to strings so they match what <select> expects
        const data = res.data.map(p => ({ ...p, id: String(p.id) }));
        setPersonas(data);
        const hasZero = data.some(p => p.id === "0");
        // 0 defaults to semantic search
        if (hasZero) setSelectedPersonaId("0");
        else if (data.length > 0) setSelectedPersonaId(data[0].id);
      })
      .catch(err => {
        console.error(err);
        setError("Could not load personas. Is the backend running?");
      });
  }, []);

  // build the dropdown list, optionally adding the temporary random user entry
  const dropdownPersonas = useMemo(() => {
    // ensure ids are strings (defensive, since personas already set them to strings)
    const base = personas.map(p => ({ ...p, id: String(p.id) }));

    // if we don't have a temp user, just return the backend list
    if (!tempUser) return base;

    // avoid duplicates if the temp user id is already in the list
    const exists = base.some(p => p.id === tempUser.id);
    if (exists) return base;

    // append the temp user option so it appears in the dropdown
    return [...base, tempUser];
  }, [personas, tempUser]);

  useEffect(() => {
    // if we never created a temp user, nothing to clean up
    if (!tempUser) return;

    // if the selected option is no longer the temp user, remove the temp user entry
    if (selectedPersonaId !== tempUser.id) {
      setTempUser(null);
    }
  }, [selectedPersonaId, tempUser]);

  const handleSearch = async (e) => {
    // stop the form submit from refreshing the page
    e.preventDefault();

    // show loading state and clear any old error
    setLoading(true);
    setError("");

    try {
      // helpful debug log to confirm what ID is actually being sent
      console.log("Sending user_id:", selectedPersonaId);

      // send request to backend recommender
      const res = await axios.post('http://localhost:8000/recommend', {
        // backend expects an int user_id, so parse the string to an int
        user_id: Number.parseInt(selectedPersonaId, 10),
        query: query
      });

      // backend returns a list of movie objects
      setMovies(res.data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch recommendations.");
    }
    setLoading(false);
  };

  const handleRandomUser = async () => {
    setError("");

    try {
      // ask backend for a valid userId from the CF trainset
      const res = await axios.get("http://localhost:8000/random_user");
      const randomId = String(res.data.user_id);
      setTempUser({
        id: randomId,
        label: `User ${randomId}`,
        description: "Real user: Hybrid (semantic retrieval + CF re-rank)."
      });

      // select the random user immediately
      setSelectedPersonaId(randomId);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch a random user.");
    }
  };

  // find the currently selected persona so we can show its description under the controls
  const currentPersona = dropdownPersonas.find(p => p.id === String(selectedPersonaId));

  // numeric id used to label the mode correctly (Mode vs Archetype vs User)
  const currentIdNum = Number.parseInt(selectedPersonaId, 10);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Recommender</h1>
      </header>

      <div className="container">
        <div className="controls">
          <div className="input-group">
            <label>Mode:</label>

            <div className="persona-row">
              <select
                className="persona-select"
                value={selectedPersonaId}
                onChange={(e) => setSelectedPersonaId(e.target.value)}
              >
                {dropdownPersonas.map(p => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>

              <button
                className="persona-random-btn"
                type="button"
                onClick={handleRandomUser}
                disabled={loading}
              >
                Random User
              </button>
            </div>
          </div>

          <form
            onSubmit={handleSearch}
            className="search-form"
          >
            <div className="search-row">
              <input
                className="search-input"
                type="text"
                placeholder="Describe the movie (e.g. '90s thriller with a twist')..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />

              <button
                className="search-button"
                type="submit"
                disabled={loading || !query.trim()}
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </div>
          </form>

          <div style={{ marginTop: 12, color: "rgba(255,255,255,0.68)", fontSize: 13 }}>
            How this works:{" "}
            <a
              href="https://github.com/davidmilanov-prog/MovieRecommendationSite#readme"
              target="_blank"
              rel="noreferrer"
            >
              Read the project README
            </a>
          </div>
        </div>

        {/* show persona description under the controls */}
        {currentPersona && (
          <div className="persona-info">
            <span className="persona-label">
              {currentIdNum >= 1000000 ? "Archetype" : currentIdNum === 0 ? "Mode" : "User"}
            </span>
            <span className="persona-desc">{currentPersona.description}</span>
          </div>
        )}

        {/* show error text if something failed */}
        {error && <div className="error">{error}</div>}

        {/* render movie results */}
        <div className="results">
          {movies.map(movie => (
            <div key={movie.movieId} className="movie-card">
              <div className="poster-wrapper">
                {/* TMDB poster path is optional, so show a placeholder if missing */}
                {movie.poster_path ? (
                  <img
                    src={`https://image.tmdb.org/t/p/w200${movie.poster_path}`}
                    alt={movie.title}
                    className="poster-img"
                  />
                ) : (
                  <div className="no-poster">No Image</div>
                )}
              </div>

              <div className="card-content">
                <div className="movie-header">
                  <h3>{movie.title}</h3>
                  <span className={`badge ${movie.match_score}`}>
                    {movie.match_score}
                  </span>
                </div>

                <p className="overview">{movie.overview}</p>

                <div className="meta">
                  <span>
                    {movie.score_label}: <strong>{movie.score}</strong>
                  </span>
                  <span>👍 Votes: {movie.votes}</span>
                </div>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}

export default App;
