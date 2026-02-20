import { Routes, Route } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import GeneratePage from "./pages/GeneratePage";
import TrainingPage from "./pages/TrainingPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<GeneratePage />} />
      <Route path="/training" element={<TrainingPage />} />
      <Route path="/chat" element={<ChatPage />} />
    </Routes>
  );
}
