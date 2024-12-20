import React from "react";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">ML Chatbot</div>
      <ul className="navbar-links">
        <li><a href="#home">Home</a></li>
        <li><a href="#features">Features</a></li>
        <li><a href="#about">About</a></li>
      </ul>
    </nav>
  );
}

export default Navbar;
