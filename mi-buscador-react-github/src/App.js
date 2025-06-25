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
      const response = await axios.get(API_URL, {
        params: { query: query }
      });

      // Verificar si la respuesta es realmente un array antes de actualizar el estado.
      if (response.data && Array.isArray(response.data)) {
        setResults(response.data);
      } else {
        // Si la API devuelve algo que no es un array (un objeto, null, etc.),
        // se trata como un error para no romper la aplicación.
        console.error("La respuesta de la API no es un array:", response.data);
        setError("Se recibió una respuesta inesperada del servidor.");
        setResults([]); // Asegurar que 'results' siga siendo un array vacío.
      }
      
    } catch (err) {
      // 3. MANEJO DE ERRORES DE RED O HTTP
      console.error("Error en la llamada a la API:", err);
      setError(err.response?.data?.error || "Error al conectar con el servidor de búsqueda.");
      setResults([]); // Asegurar que 'results' sea un array.
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Buscador de Issues Similares (PowerToys)</h1>

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
            
            <div className="metadata-grid">
                {/* El Ticket ID ahora es un enlace */}
                <span>
                  <strong>Ticket ID:</strong>{' '}
                  <a 
                    href={`https://github.com/microsoft/PowerToys/issues/${result.original_ticket_id}`} 
                    target="_blank" 
                    rel="noopener noreferrer"
                  >
                    #{result.original_ticket_id}
                  </a>
                </span>
                
                <span><strong>Score:</strong> {result.score.toFixed(4)}</span>
                <span><strong>Área:</strong> {result.area}</span>
                <span><strong>Versión:</strong> {result.powertoys_version}</span>
            </div>

            {result.tags && Array.isArray(result.tags) && result.tags.length > 0 && (
              <div className="tags">
                <strong>Tags:</strong>
                {result.tags.map((tag, i) => (
                  <span key={i} className="tag-item">{tag}</span>
                ))}
              </div>
            )}
            
            <div className="behavior-section">
                <h4>Comportamiento Actual</h4>
                <p>{result.actual_behavior || "No detallado."}</p>
            </div>
            
            <div className="solution-section">
                {/* El ID de la fuente de la solución ahora también es un enlace */}
                <h4>
                  Solución Encontrada{' '}
                  {result.answer_source_id && result.original_ticket_id !== result.answer_source_id ? (
                    <span>
                      (de issue{' '}
                      <a 
                        href={`https://github.com/microsoft/PowerToys/issues/${result.answer_source_id}`} 
                        target="_blank" 
                        rel="noopener noreferrer"
                      >
                        #{result.answer_source_id}
                      </a>)
                    </span>
                  ) : '(Respuesta Directa)'}
                </h4>
                <pre className="answer-box">{result.answer}</pre>
            </div>

          </div>
        ))}
        {!isLoading && results.length === 0 && searchedQuery && !error && (
          <p className="no-results-message">No se encontraron resultados similares.</p>
        )}
      </div>
    </div>
  );
}

export default App;