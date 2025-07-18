"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { LogOut, Upload, ShieldCheck, AlertTriangle, CheckCircle, Loader } from "lucide-react";
import { SHA256 } from 'crypto-js';

// Header Component
const AdminHeader = ({ onLogout }) => (
  <header className="bg-gray-800 text-white px-6 py-4 flex justify-between items-center shadow-md">
    <div className="flex items-center space-x-4">
      <ShieldCheck className="w-8 h-8" />
      <h1 className="text-xl font-bold">Admin Dashboard</h1>
    </div>
    <Button 
      variant="ghost" 
      className="text-white hover:bg-gray-700 cursor-pointer" 
      onClick={onLogout}
    >
      <LogOut className="mr-2 h-4 w-4" /> Logout
    </Button>
  </header>
);

// Footer Component
const AdminFooter = () => (
  <footer className="bg-gray-800 text-white px-6 py-4 text-center">
    <p className="text-sm">© {new Date().getFullYear()} Admin Management System. All rights reserved.</p>
  </footer>
);

// Login Form Component
const LoginForm = ({ username, setUsername, password, setPassword, handleLogin, error }) => (
  <Card className="w-full max-w-md mx-auto mt-10">
    <CardHeader>
      <CardTitle className="text-center">Admin Login</CardTitle>
    </CardHeader>
    <CardContent>
      <form onSubmit={handleLogin} className="space-y-4">
        <div>
          <label htmlFor="username" className="block mb-2">Username</label>
          <Input 
            id="username"
            type="text" 
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter username"
            required
            className="w-full"
          />
        </div>
        <div>
          <label htmlFor="password" className="block mb-2">Password</label>
          <Input 
            id="password"
            type="password" 
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter password"
            required
            className="w-full"
          />
        </div>
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="ml-2">{error}</AlertDescription>
          </Alert>
        )}
        <Button type="submit" className="w-full mt-4 cursor-pointer">
          Login
        </Button>
      </form>
    </CardContent>
  </Card>
);

// Model Retraining Component
const ModelRetraining = ({ 
  selectedFile, 
  handleFileChange, 
  handleModelRetrain, 
  trainingStatus,
  isProcessing 
}) => (
  <Card className="w-full max-w-xl mx-auto mt-10">
    <CardHeader>
      <CardTitle>Model Retraining</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="space-y-6">
        <div>
          <label htmlFor="jsonFile" className="block mb-2 flex items-center">
            <Upload className="mr-2 h-5 w-5" />
            Upload Training JSON
          </label>
          <Input 
            id="jsonFile"
            type="file" 
            accept=".json"
            onChange={handleFileChange}
            className="w-full"
          />
          {selectedFile && (
            <p className="text-sm text-gray-500 mt-2">
              Selected File: {selectedFile.name}
            </p>
          )}
        </div>
        
        <div className="space-y-4">
          <Button 
            onClick={handleModelRetrain} 
            disabled={!selectedFile || isProcessing}
            className="w-full cursor-pointer"
          >
            {isProcessing ? (
              <>
                <Loader className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Retrain Model"
            )}
          </Button>
          
          {trainingStatus && !isProcessing && (
            <Alert 
              variant={trainingStatus.includes('successfully') ? 'default' : 'destructive'}
              className="mt-4"
            >
              {trainingStatus.includes('successfully') ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-red-500" />
              )}
              <AlertDescription className="ml-2">
                {trainingStatus}
              </AlertDescription>
            </Alert>
          )}
        </div>
        
        <div className="bg-gray-100 p-4 rounded-lg">
          <h3 className="font-semibold mb-2">Retraining Guidelines</h3>
          <ul className="list-disc list-inside text-sm text-gray-600">
            <li>Only JSON files are accepted</li>
            <li>Ensure the JSON structure matches the expected format</li>
            <li>Large files may take longer to process</li>
            <li>Backup your existing model before retraining</li>
          </ul>
        </div>
      </div>
    </CardContent>
  </Card>
);

// Main Admin Interface Component
const AdminInterface = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const PASSWORD = "a871b9fb4727327afce037abcf07cfd791e1d3f87f4e8e3fe83d5fcc4d33d79e";

  const handleLogin = (e) => {
    e.preventDefault();
    
    // Hash the password
    const hashHex = SHA256(password).toString();
    
    if (hashHex === PASSWORD) {
      setIsLoggedIn(true);
      setError('');
    } else {
      setError("Invalid credentials");
    }
  };
  

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUsername('');
    setPassword('');
    setSelectedFile(null);
    setTrainingStatus('');
    setIsProcessing(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    setSelectedFile(file || null);
    // Clear previous status messages when a new file is selected
    setTrainingStatus('');
  };

  const handleModelRetrain = async () => {
    if (!selectedFile) {
      setTrainingStatus("Please select a JSON file");
      return;
    }

    setIsProcessing(true);
    setTrainingStatus('');

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/retrain_examples`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setTrainingStatus("Model retrained successfully!");
      } else {
        const errorMessage = await response.text();
        setTrainingStatus(errorMessage || "Model retraining failed");
      }
    } catch (error) {
      setTrainingStatus("Error during model retraining");
      console.error("Retraining error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {isLoggedIn && <AdminHeader onLogout={handleLogout} />}
      
      <main className="flex-grow flex items-center justify-center">
        {!isLoggedIn ? (
          <LoginForm
            username={username}
            setUsername={setUsername}
            password={password}
            setPassword={setPassword}
            handleLogin={handleLogin}
            error={error}
          />
        ) : (
          <ModelRetraining
            selectedFile={selectedFile}
            handleFileChange={handleFileChange}
            handleModelRetrain={handleModelRetrain}
            trainingStatus={trainingStatus}
            isProcessing={isProcessing}
          />
        )}
      </main>
      
      {isLoggedIn && <AdminFooter />}
    </div>
  );
};

export default AdminInterface;