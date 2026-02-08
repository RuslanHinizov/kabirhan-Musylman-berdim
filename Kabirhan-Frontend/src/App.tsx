import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PublicDisplay } from './pages/PublicDisplay';
import { OperatorPanel } from './pages/OperatorPanel';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PublicDisplay />} />
        <Route path="/operator" element={<OperatorPanel />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
