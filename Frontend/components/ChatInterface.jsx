"use client";
// ChatInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Send, Trash2, User, Bot, BarChart2, FileDown, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioStream, setAudioStream] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [nextId, setNextId] = useState(0);
  const [includeChart, setIncludeChart] = useState(false);
  const API_URL = process.env.NEXT_PUBLIC_API_URL;
  
  // Example questions for the bubbles
  const exampleQuestions = [
    "Which products are the most sold overall?",
    "Which staff member has made the most sales?",
    "How many orders has each store processed?",
    "Who are the top 5 customers by total amount spent?"
  ];

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    clearChat();
  }, []);

  // Auto input focus on rendering
  useEffect(() => {
    inputRef.current.focus()
  }, [])

  useEffect(() => {
    if (messages.length > 0 && !isLoading && !isRecording) {
      inputRef.current?.focus();
    }
  }, [messages, isLoading, isRecording]);

  // Handle clicking on a question bubble
  const handleQuestionBubbleClick = (question) => {
    setInputText(question);
    // Focus the input field
    document.querySelector('input[type="text"]').focus();
  };

  const handleRegenerateResponse = async (originalMessage) => {
    // Remove any previous bot responses for this user message
    const filteredMessages = messages.filter(msg => 
      msg.sender !== 'assistant' || msg.originalMessageId !== originalMessage.id
    );

    // Set messages and trigger sending the original message again
    setMessages(filteredMessages);
    setInputText(originalMessage.text);
    
    // Simulate sending the message programmatically
    await handleSendMessage(originalMessage.text);
  };

  // Handle sending text message
  const handleSendMessage = async (manualText = null) => {

    const textToSend = manualText || inputText;

    if (!textToSend.trim()) return;
    
    // Add user message to chat
    const userMessage = { 
      id: Date.now(), 
      text: textToSend, 
      sender: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    
    try {
        // Add user message to the messages array first
        const updatedMessages = [...messages, userMessage];
        
        console.log(updatedMessages);

        let responseText = "";
        // Create bot message object once
        var botMessage = { 
          id: nextId, 
          text: responseText, 
          sender: 'assistant',
          hasExcel: false,
          chartUrl: null,
          hasPdf: false,
          pdfData: null,
          excelData: null,
          timestamp: new Date().toISOString()
        };
        setNextId(nextId + 1);
        // Check if the input contains PDF-related keywords
        const shouldGeneratePdf = containsPdfKeywords(userMessage.text);
        if (shouldGeneratePdf) {
          botMessage.hasPdf = true;
          const pdfData = await generatePdf(userMessage.text)
          botMessage.pdfData = pdfData;
          setMessages([...updatedMessages, botMessage]);
        }
        else {
          // Send the API request with the updated messages
          const response = await queryApi(updatedMessages);
      
          if (response) {
            // Process the streaming response from the API
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                responseText += decoder.decode(value, { stream: true });
                
                // Just update the text field
                botMessage.text = responseText;
                
                // Update messages with the current state of botMessage
                setMessages([...updatedMessages, botMessage]);
              }
              // If chart is requested, generate it after the response is complete
              if (includeChart) {
                await generateChart(updatedMessages, botMessage);
              }
              // Check if Excel is available and update the hasExcel field
              const hasExcel = response.headers.get("X-Has-Excel") === "true";
              if (hasExcel) {
                botMessage.hasExcel = true;
                const excelData = await generateExcel();
                botMessage.excelData = excelData;
                // Final update with Excel flag
                setMessages([...updatedMessages, botMessage]);
              }

        }     
        else {
          throw new Error("Failed to get response from API");
        }
      }
    
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages(prev => [...prev, { 
        id: Date.now() + 1, 
        text: "Sorry, I couldn't process this due to this error:" + error, 
        sender: 'assistant',
        error: true,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
      setIncludeChart(false);
      }
  };

  // Function to generate chart after response is complete
  const generateChart = async (updatedMessages, botMessage, question) => {
  try {
      const response = await fetch(`${API_URL}/generate_chart/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'text/plain'
        },
        body: question
      });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to generate chart');
    }
    
    const data = await response.json();
    
    // Update the existing bot message with chart information
    botMessage.chartUrl = data.chart;
    botMessage.text += "\n\n📊 Here is a visualization of the analyzed data:";
    
    // Update the messages
    setMessages([...updatedMessages, botMessage]);
    
  } catch (error) {
    console.error('Error generating chart:', error);
    // Add error message about chart generation
    const chartErrorMessage = {
      id: Date.now(), 
      text: `Sorry, I couldn't generate the chart: ${error.message}`, 
      sender: 'assistant',
      error: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, chartErrorMessage]);
  } finally {
    setIncludeChart(false);
  } 
  };

  // Send query to API
  const queryApi = async (messagesForApi) => {
    try {
      const payload = messagesForApi.map(msg => ({
        role: msg.sender,
        content: msg.text
      }));
      
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      
      return response;
    } catch (error) {
      console.error('API Error:', error);
      return null;
    }
  };

  // Function to generate excel
  const generateExcel = async () => {
    try {
      const response = await fetch(`${API_URL}/api/excel/`);
      if (!response.ok) {
        throw new Error(`Failed to generate PDF: ${response.status}`);
      }
      const blob = await response.blob();
      return blob;
      } catch (error) {
        console.error('Error generating Excel:', error);
        return null;
      }
  };
  // Download excel
  const downloadExcel = async (excelData) => {
    try {
        const url = window.URL.createObjectURL(excelData);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'data.xlsx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
      } catch (error) {
      console.error('Error downloading Excel:', error);
      return false;
    }
  };

    // Function to check if user input contains PDF-related keywords
    const containsPdfKeywords = (text) => {
      const keywords = ["generate", "pdf", "report"];
  
      // Check if any of the keywords are present in the text
      return keywords.some(keyword => text.toLowerCase().includes(keyword.toLowerCase()));
    };
  
    // Function to generate PDF
    const generatePdf = async (question) => {
      try {
        const response = await fetch(`${API_URL}/generate_pdf/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'text/plain'
          },
          body: question
        });

        if (!response.ok) {
          const errorMessage = await response.text();
          throw new Error(errorMessage);
        }

        const blob = await response.blob();
        return blob;
      } catch (error) {
        console.error('Error generating PDF:', error);
        throw error;
      }
    };

    const downloadPdf = async(pdf) => {
      try{
        const url = window.URL.createObjectURL(pdf);
        const a = document.createElement('a');
        a.href = url;
        a.download = `repport-${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        return true;
      }catch (error) {
        console.error('Error generating PDF:', error);
        throw error;
      }
    };

  // Handle starting audio recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = handleAudioStop;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert("Please allow access to microphone.");
    }
  };

  // Handle stopping audio recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      audioStream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  // Process audio after recording stops
  const handleAudioStop = async () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
    setIsLoading(true);
    try {
      // Send audio to the transcription API
      const transcribedText = await transcribeAudio(audioBlob);
      
      if (transcribedText) {
        // Add transcribed message to chat
        const userMessage = { 
          id: Date.now(), 
          text: transcribedText, 
          sender: 'user',
          isAudio: true,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, userMessage]);        
        // Send to API and get response
        const updatedMessages = [...messages, userMessage];
        let responseText = "";
          
        // Create bot message object once
        var botMessage = { 
          id: nextId, 
          text: responseText, 
          sender: 'assistant',
          hasExcel: false,
          chartUrl: null,
          hasPdf: false,
          pdfData: null,
          excelData: null,
          timestamp: new Date().toISOString()
        };
        setNextId(nextId + 1);
        // Check if the input contains PDF-related keywords
        const shouldGeneratePdf = containsPdfKeywords(userMessage.text);
        if (shouldGeneratePdf) {
          botMessage.hasPdf = true;
          const pdfData = await generatePdf(userMessage.text)
          botMessage.pdfData = pdfData;
          setMessages([...updatedMessages, botMessage]);
        }
        else {
          // Send the API request with the updated messages
          const response = await queryApi(updatedMessages);
      
          if (response) {
            // Process the streaming response from the API
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                responseText += decoder.decode(value, { stream: true });
                
                // Just update the text field
                botMessage.text = responseText;
                
                // Update messages with the current state of botMessage
                setMessages([...updatedMessages, botMessage]);
              }
              // If chart is requested, generate it after the response is complete
              if (includeChart) {
                await generateChart(updatedMessages, botMessage);
              }
              // Check if Excel is available and update the hasExcel field
              const hasExcel = response.headers.get("X-Has-Excel") === "true";
              if (hasExcel) {
                botMessage.hasExcel = true;
                const excelData = await generateExcel();
                botMessage.excelData = excelData;
                // Final update with Excel flag
                setMessages([...updatedMessages, botMessage]);
              }
          }
        }
      } else {
        throw new Error("Transcription failed");
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      setMessages(prev => [...prev, { 
        id: Date.now() + 1, 
        text: "Sorry, I couldn't process the audio. Please try again or type your message instead.", 
        sender: 'assistant',
        error: true,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
      setIncludeChart(false); // Reset chart inclusion state
    }
  };

  // Send audio to transcription API
  const transcribeAudio = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recorded_audio.wav');
      
      const response = await fetch(`${API_URL}/transcribe/`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        return result.transcription || "";
      } else {
        console.error(`Transcription error: ${response.status}`);
        return "";
      }
    } catch (error) {
      console.error('Error during transcription:', error);
      return "";
    }
  };

  // Clear all chat messages
  const clearChat = async() => {
    await fetch(`${API_URL}/clear_messages/`);
    setNextId(0);
    setMessages([]);
    setIncludeChart(false);
  };

  return (    
    <div className="flex flex-col h-screen w-full bg-gray-50">
      {/* Chat header */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white px-6 py-4 shadow-md">
        <div className="flex justify-between items-center max-w-7xl mx-auto">
          <div className="flex items-center space-x-3">
            <Bot size={24} className="text-blue-200" />
            <h1 className="text-2xl font-semibold tracking-wide">DeepInsight Chatbot</h1>
          </div>
          <div className="flex items-center">
            <button
              onClick={clearChat}
              className="flex items-center space-x-2 bg-red-600 bg-opacity-30 hover:bg-red-800 px-3 py-1.5 rounded-lg transition-all duration-200 hover:shadow-lg cursor-pointer"
              aria-label="Clear chat"
            >
              <Trash2 size={16} />
              <span className="text-sm font-medium">Clear Chat</span>
            </button>
          </div>
        </div>
      </div>
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto bg-white bg-opacity-70">
        <div className="max-w-5xl mx-auto p-4 md:p-6 h-full">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500 py-16">
              <Bot size={48} className="text-gray-300 mb-4" />
              <p className="text-lg mb-8">Start a conversation by typing a message or using your microphone</p>
              <br />
              {/* Question bubbles */}
              <div className="w-full max-w-3xl">
              <p className="text-sm text-gray-500 mb-3 text-center">Examples of questions:</p>
              <div className="flex flex-wrap justify-center gap-3">
                {exampleQuestions.map((question, index) => {
                  // Array of color combinations (background, hover background, text, border)
                  const colorSchemes = [
                    "bg-blue-50 hover:bg-blue-100 text-blue-700 border-blue-200",
                    "bg-green-50 hover:bg-green-100 text-green-700 border-green-200",
                    "bg-purple-50 hover:bg-purple-100 text-purple-700 border-purple-200",
                    "bg-amber-50 hover:bg-amber-100 text-amber-700 border-amber-200",
                    "bg-rose-50 hover:bg-rose-100 text-rose-700 border-rose-200"
                  ];
                  
                  // Select a color scheme based on index (cycles through the colors)
                  const colorScheme = colorSchemes[index % colorSchemes.length];
                  
                  return (
                    <button
                      key={index}
                      onClick={() => handleQuestionBubbleClick(question)}
                      className={`px-6 py-3 ${colorScheme} 
                                border rounded-full text-sm font-medium
                                transition-all duration-150 hover:shadow-md 
                                cursor-pointer transform hover:scale-105 
                                whitespace-nowrap overflow-hidden text-ellipsis max-w-[220px]`}
                    >
                      {question}
                    </button>
                  );
                })}
              </div>
            </div>
            </div>
          ) : (
            messages.map((message) => (
              <div 
                key={message.id} 
                className={`mb-6 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex max-w-xl ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`flex-shrink-0 flex items-start mt-1 ${message.sender === 'user' ? 'ml-3' : 'mr-3'}`}>
                    <div className={`rounded-full p-2 ${message.sender === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                      {message.sender === 'user' ? (
                        <User size={20} className="text-blue-600" />
                      ) : (
                        <Bot size={20} className="text-indigo-600" />
                      )}
                    </div>
                  </div>
                  <div>
                    <div 
                      className={`px-4 py-2.5 rounded-2xl ${
                        message.sender === 'user' 
                          ? 'bg-gray-200 text-black' 
                          : 'bg-white-100 text-gray-800'
                      } ${message.error ? 'bg-red-100 text-red-800' : ''}`}
                    >
                      {message.isAudio && <span className="text-xs block mb-1 opacity-75">🎤 Audio transcript</span>}
                      <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                              table: ({ node, ...props }) => (
                                  <table className="min-w-full border border-gray-300 my-2">{props.children}</table>
                              ),
                              th: ({ children }) => <th className="px-4 py-2 bg-gray-200 border">{children}</th>,
                              td: ({ children }) => <td className="px-4 py-2 border">{children}</td>,
                      }}>
                          {message.text}
                      </ReactMarkdown>
                      {message.chartUrl && (
                        <div className="mt-3 border border-gray-200 rounded-lg overflow-hidden">
                          <img 
                            src={message.chartUrl} 
                            alt="Data Chart" 
                            className="w-full object-contain" 
                            style={{ maxHeight: '300px' }}
                          />
                        </div>
                      )}
                      {message.sender === "assistant" && message.hasExcel && (
                        <>
                        <p className="mt-2">
                          {message.text === "" || !message.text 
                            ? "Your report has been successfully generated! You can view it below."
                            : "Please note that the displayed results are only a sample of the returned data. To view the complete results, please refer to the attached Excel file."}
                        </p>
                          <button
                            onClick={() => downloadExcel(message.excelData)}
                            className="flex items-center space-x-2 bg-white border border-gray-300 px-4 py-2 rounded-lg transition-all duration-200 hover:bg-gray-50 cursor-pointer mr-4 mt-1"
                            aria-label="Télécharger Excel"
                          >
                            <span className="text-sm font-medium text-gray-700">📥 Download Excel Results</span>
                          </button>
                        </>
                      )}
                      {message.sender === "assistant" && message.hasPdf && (
                        <>
                          <p className="mt-2">
                            {"A PDF report for your order has been successfully generated. You can download it and view all its information."}
                          </p>
                          <button
                            onClick={() => downloadPdf(message.pdfData)}
                            className="flex items-center space-x-2 bg-white border border-red-300 px-4 py-2 rounded-lg transition-all duration-200 hover:bg-red-50 cursor-pointer mr-4 mt-1"
                            aria-label="Télécharger PDF"
                          >
                            <FileDown size={16} className="text-red-600" />
                            <span className="text-sm font-medium text-red-600">Download PDF Report</span>
                          </button>
                        </>
                      )}
                    </div>
                         {message.sender === 'user' && <button
                          onClick={() => handleRegenerateResponse(message)}
                          className="mr-2 p-1.5 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors duration-200 mr-2"
                          aria-label="Regenerate response"
                          disabled={isLoading}
                        >
                          <RefreshCw size={12} className="text-gray-600" />
                        </button>}
                    <div className={`text-xs text-gray-500 mt-1 ${message.sender === 'user' ? 'text-right' : 'text-left pl-3'}`}>
                      {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex mb-6">
              <div className="flex-shrink-0 flex items-start mt-1 mr-3">
                <div className="rounded-full p-2 bg-gray-100">
                  <Bot size={20} className="text-indigo-600" />
                </div>
              </div>
              <div 
                className="px-4 py-2.5 bg-gray-100 text-gray-800 rounded-2xl"
              >
                <div className="flex space-x-2">
                  <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '600ms' }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      {/* Input area */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`p-2.5 rounded-full mr-3 ${
                isRecording 
                  ? 'bg-red-100 text-red-600 hover:bg-red-200' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              } transition-colors hover:shadow-md cursor-pointer`}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
              disabled={isLoading}
            >
              {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
            </button>
            
            {/* Chart toggle button */}
            <button
              onClick={() => setIncludeChart(!includeChart)}
              className={`p-2.5 rounded-full mr-3 ${
                includeChart 
                  ? 'bg-purple-100 text-purple-600 hover:bg-purple-200' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              } transition-colors hover:shadow-md cursor-pointer`}
              aria-label="Include chart with response"
              disabled={isLoading || isRecording}
              title="Include chart with response"
            >
              <BarChart2 size={20} />
            </button>
              <div className="flex-1 flex items-center">
                <input
                  type="text"
                  ref={inputRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder={"Type your message..."}
                  className={`flex-1 py-2.5 px-4 rounded-full focus:outline-none transition-all ${
                    includeChart ? 'bg-purple-50 border border-purple-200' : 'bg-gray-100'
                  }`}
                  disabled={isRecording || isLoading}
                />
              </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isLoading}
              className="p-2.5 bg-blue-600 text-white rounded-full ml-3 hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 hover:shadow-lg cursor-pointer"
              aria-label="Send message"
            >
              <Send size={20} />
            </button>
          </div>
          {isRecording && (
            <div className="mt-2 text-center text-sm text-red-500 animate-pulse">
              Recording... Click the microphone icon to stop.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;