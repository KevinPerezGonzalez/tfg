// src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // archivo para estilos

const API_URL = 'http://localhost:5000/api/search'; // URL de backend Flask

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchedQuery, setSearchedQuery] = useState('');

  const handleInputChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Por favor, introduce un término de búsqueda.");
      setResults([]);
      setSearchedQuery('');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults([]); // Limpia resultados anteriores
    setSearchedQuery(query); // Guarda la consulta que se está buscando

    try {
      // Asegurar de que el servidor Flask (app.py) está corriendo
      const response = await axios.get(API_URL, {
        params: { query: query }
      });
      setResults(response.data);
    } catch (err) {
      console.error("Error fetching search results:", err);
      setError(err.response?.data?.error || "Error al conectar con el servidor de búsqueda. Asegúrate de que está activo.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Buscador de Casos de Soporte Similares (React)</h1>

      <form onSubmit={handleSubmit} className="search-form">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Describe el problema aquí..."
          className="search-input"
        />
        <button type="submit" disabled={isLoading} className="search-button">
          {isLoading ? 'Buscando...' : 'Buscar'}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}

      {searchedQuery && !isLoading && (
        <h2>Resultados para: "{searchedQuery}"</h2>
      )}

      <div className="results-area">
        {isLoading && <p className="loading-message">Cargando resultados...</p>}

        {!isLoading && results.length > 0 && results.map((result, index) => (
          <div key={result.doc_id_internal || index} className="result-item">
            <h3>{result.subject}</h3>
            <p>
              <strong>ID Original:</strong> {result.original_ticket_id} | 
              <strong> ID Interno:</strong> {result.doc_id_internal} | 
              <strong> Score:</strong> {result.score.toFixed(4)}
            </p>
            {result.tags && result.tags.length > 0 && (
              <p className="tags">
                <strong>Tags:</strong>
                {result.tags.map((tag, i) => (
                  <span key={i} className="tag-item">{tag}</span>
                ))}
              </p>
            )}
            <p><strong>Respuesta / Solución:</strong><br />{result.answer}</p>
          </div>
        ))}
        {!isLoading && results.length === 0 && searchedQuery && !error && (
          <p className="no-results-message">No se encontraron resultados similares para tu consulta.</p>
        )}
      </div>
    </div>
  );
}

export default App;