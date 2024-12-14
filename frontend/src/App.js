import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const apiUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
  const [query, setQuery] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedOption, setSelectedOption] = useState("");
  const [results, setResults] = useState([]);
  const [models, setModels] = useState(["bert", "minilm", "roberta", "sbert"]);
  const [options, setOptions] = useState(["0", "1", "2", "3", "4", "5"]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastQueries, setLastQueries] = useState([]);

  useEffect(() => {
    // Fetch last 10 queries including model and option
    const fetchLastQueries = async () => {
      try {
        const response = await axios.get("http://localhost:8000/last_queries");
        setLastQueries(response.data);
      } catch (error) {
        console.error("Error fetching last queries:", error);
      }
    };

    fetchLastQueries();
  }, []);

  // Handle the search action
  const handleSearch = async () => {
    if (!query || !selectedModel || !selectedOption) {
      setError("Please enter a query and select a model and option.");
      return;
    }

    setIsLoading(true);
    setError(""); // Clear previous errors
    try {
      const response = await axios.post("http://localhost:8000/search", {
        query,
        model_name: selectedModel,
        option: selectedOption,
      });
      setResults(response.data.results);

      // Reorder the queries and update the list
      setLastQueries((prevQueries) => {
        const updatedQueries = [
          { query, model: selectedModel, option: selectedOption },
          ...prevQueries.filter(
            (item) =>
              item.query !== query ||
              item.model !== selectedModel ||
              item.option !== selectedOption
          ),
        ];
        return updatedQueries.slice(0, 10); // Keep only the last 10 queries
      });
    } catch (err) {
      console.error("Error during search:", err);
      setError("An error occurred during the search.");
      setTimeout(() => setError(""), 5000);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission with Enter key
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleSearch();
    }
  };

  // Handle clicking on a previous query to perform a search
  const handleClickPreviousQuery = (queryText, model, option) => {
    setQuery(queryText);
    setSelectedModel(model);
    setSelectedOption(option);
    handleSearch();
  };

  return (
    <div className="flex flex-row h-screen">
      {/* Left-side Table for Last 10 Queries */}
      <div className="w-full md:w-1/4 p-8 overflow-y-auto bg-gray-100">
        <h3 className="font-bold mb-4">Last 10 Queries</h3>
        <table className="table-auto w-full">
          <thead>
            <tr>
              <th className="border p-2">Query</th>
              <th className="border p-2">Model</th>
              <th className="border p-2">Option</th>
            </tr>
          </thead>
          <tbody>
            {lastQueries.map((queryData, index) => (
              <tr
                key={index}
                className="cursor-pointer hover:bg-gray-200"
                onClick={() =>
                  handleClickPreviousQuery(
                    queryData.query,
                    queryData.model,
                    queryData.option
                  )
                }
              >
                <td className="border p-2">{queryData.query}</td>
                <td className="border p-2">{queryData.model}</td>
                <td className="border p-2">{queryData.option}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Right-side Content */}
      <div className="flex-1 p-8 overflow-y-auto flex flex-col justify-start items-center">
        {/* Search Form (Centered) */}
        <div
          className={`flex gap-4 mb-4 w-full md:w-1/2 ${
            isLoading ? "pointer-events-none opacity-50" : ""
          }`}
        >
          <input
            type="text"
            placeholder="Enter your query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyPress}
            className="border rounded-lg px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400"
            disabled={isLoading}
          />
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="border rounded-lg px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400"
            disabled={isLoading}
          >
            <option value="">Select a model</option>
            {models.map((model, index) => (
              <option key={index} value={model}>
                {model}
              </option>
            ))}
          </select>
          <select
            value={selectedOption}
            onChange={(e) => setSelectedOption(e.target.value)}
            className="border rounded-lg px-4 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400"
            disabled={isLoading}
          >
            <option value="">Select an option</option>
            {options.map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </select>
          <button
            onClick={handleSearch}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400"
            disabled={isLoading}
          >
            Search
          </button>
        </div>

        {/* Display loading spinner */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-300 bg-opacity-70 z-50">
            <div className="animate-spin rounded-full h-24 w-24 border-t-4 border-blue-500"></div>
          </div>
        )}

        {/* Error Popup */}
        {error && (
          <div className="absolute bottom-0 right-0 m-4 bg-red-500 text-white px-4 py-2 rounded-md shadow-lg">
            {error}
          </div>
        )}

        {/* Display search results */}
        {results.length > 0 && (
          <div className="w-full max-w-3xl mt-4">
            <ul className="bg-white shadow rounded-lg p-4">
              {results.map((result, index) => (
                <li
                  key={index}
                  className="border-b last:border-b-0 py-2 flex justify-between"
                >
                  <span className="font-bold">
                    {result.instagram_data?.full_name ||
                      result.instagram_data?.name}
                  </span>
                  <span>
                    <strong>Distance:</strong> {result.distance.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
