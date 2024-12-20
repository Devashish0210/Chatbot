import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Register from "./components/Register";
import Login from "./components/Login";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isRegistering, setIsRegistering] = useState(true);

  const handleRegisterSuccess = () => {
    setIsRegistering(false);
  };

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
  };

  if (!isAuthenticated) {
    return isRegistering ? (
      <Register onRegisterSuccess={handleRegisterSuccess} />
    ) : (
      <Login onLoginSuccess={handleLoginSuccess} />
    );
  }

  return (
    <div className="app-container">
      <Navbar />
      <ChatInterface />
    </div>
  );
}

export default App;
