import React, { useState } from 'react';


function ChatInput({ onSendMessage, isLoading }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <div
      style={{
       
        height: '100vh',
        width: '100vw',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <form
        onSubmit={handleSubmit}
        className="chat-input-form"
        style={{
         
          padding: '20px',
          borderRadius: '12px',
          display: 'flex',
          gap: '10px',
          alignItems: 'center',
          width: '90%',
          maxWidth: '600px',
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about real estate (e.g., Analyze Wakad)"
          disabled={isLoading}
          style={{
            flex: 1,
            padding: '12px 15px',
            borderRadius: '8px',
            border: '1px solid #ccc',
            fontSize: '1rem',
          }}
        />
        <button
          type="submit"
          disabled={isLoading}
          style={{
            padding: '12px 20px',
            backgroundColor: '#05445E',
            color: '#ffffff',
            border: 'none',
            borderRadius: '8px',
            fontWeight: 'bold',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.3s',
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default ChatInput;
