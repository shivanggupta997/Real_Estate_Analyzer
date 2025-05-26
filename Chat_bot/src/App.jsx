import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

import ChatInput from './Components/ChatInput';
import ChatMessage from './Components/ChatMessage';


// const API_URL = 'http://localhost:8000/api/analyze/';
// In Chat_bot/src/App.jsx
const API_URL = process.env.VITE_APP_API_URL || 'http://localhost:8000/api/analyze/'; 

function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! How can I help you with real estate analysis today?' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = async (userInput) => {
    const newMessages = [...messages, { sender: 'user', text: userInput }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      const response = await axios.post(API_URL, { query: userInput });
      const botResponse = {
        sender: 'bot',
        analysis: response.data 
      };
      setMessages([...newMessages, botResponse]);
    } catch (error) {
      console.error("Error fetching analysis:", error);
      const errorResponse = {
        sender: 'bot',
        text: "Sorry, I encountered an error. Please try again."
      };
      setMessages([...newMessages, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App"
     style={{
   backgroundImage: 'url("https://t3.ftcdn.net/jpg/01/41/14/76/360_F_141147632_revBO3mgrWS2Y6KqeyYT8J87XFFpqs7X.jpg")',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    boxSizing: 'border-box',
  }}>
    <h1 style={{ color: 'white', fontFamily: 'Poppins, sans-serif' }}>
  🏛️REAL ESTATE ANALYZER🏛️
</h1>

      <div className="chat-container" ref={chatContainerRef}>
        {messages.map((msg, index) => (
          <ChatMessage key={index} message={msg} />
        ))}
        {isLoading && <div className="message bot loading-message">Analyzing...</div>}
      </div>
      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
     
    </div>
  );
}

export default App;