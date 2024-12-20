import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "../App.css";
import { useLayoutEffect } from "react";

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const chatWindowRef = useRef(null);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (input.trim() === "") return;
  
    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
  
    try {
      const response = await axios.post(
        "http://localhost:8000/chat",
        { text: input },
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
  
      const botMessage = { text: response.data.response, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error.response?.data || error.message);
    } finally {
      setIsTyping(false);
    }
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const filePreviewUrl = URL.createObjectURL(file);
    setPreviewImage(filePreviewUrl);

    const userMessage = { text: `Image uploaded: ${file.name}`, sender: "user", image: filePreviewUrl, };
    setMessages((prev) => [...prev, userMessage]);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("text", "");

    setIsTyping(true);

    try {
      const response = await axios.post("http://localhost:8000/chat", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const botMessage = { text: response.data.response, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="container">
      <div className="header">ML Chatbot</div>
      <div className="chat-window">
  {messages.map((msg, index) => (
    <div
      key={index}
      className={`message ${msg.sender === "bot" ? "bot" : "user"}`}
    >
      {msg.sender === "bot" && (
        <img
          className="avatar"
          src="bot-avatar.png" 
          alt="Bot"
        />
      )}
      {msg.sender === "user" && (
        <img
          className="avatar"
          src="user-avatar.png" 
          alt="User"
        />
      )}

      <div className="message-content">
        {msg.text}
        {msg.image && (
          <img
            src={msg.image}
            alt="Uploaded Preview"
            style={{
              maxWidth: "100%", 
              maxHeight: "250px", 
              marginTop: "10px", 
              borderRadius: "8px", 
              border: "1px solid #ddd", 
              display: "block", 
            }}
          />
        )}
        <div className="timestamp">
          {new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>
      </div>
    </div>
  ))}
</div>
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          className="input-box"
        />
        <label htmlFor="file-upload" className="custom-file-upload">
          Attach Image
        </label>
        <input
          id="file-upload"
          type="file"
          onChange={handleImageUpload}
          className="file-input"
        />
        <button onClick={sendMessage} className="send-button">
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatInterface;