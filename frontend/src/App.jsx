import React, { useState } from 'react';
import Sidebar from './components/layout/Sidebar';
import Topbar from './components/layout/Topbar';
import Overview from './components/dashboard/Overview';
import StudentDirectory from './components/dashboard/StudentDirectory';
import AttendanceLogs from './components/dashboard/AttendanceLogs';
import FaceReview from './components/dashboard/FaceReview';
import LiveCameras from './components/dashboard/LiveCameras';
import Settings from './components/dashboard/Settings';
import SystemHealth from './components/dashboard/SystemHealth';

function App() {
  const [activePage, setActivePage] = useState('overview');

  const getTitle = () => {
    switch (activePage) {
      case 'overview': return 'Dashboard Overview';
      case 'students': return 'Student Directory';
      case 'attendance': return 'Attendance Logs';
      case 'review': return 'Unknown Face Review';
      case 'cameras': return 'Live Camera Feeds';
      case 'health': return 'System Health Status';
      case 'settings': return 'System Settings';
      default: return 'EMSVIA';
    }
  };

  const renderContent = () => {
    switch (activePage) {
      case 'overview': return <Overview />;
      case 'students': return <StudentDirectory />;
      case 'attendance': return <AttendanceLogs />;
      case 'review': return <FaceReview />;
      case 'cameras': return <LiveCameras />;
      case 'health': return <SystemHealth />;
      case 'settings': return <Settings />;
      default:
        return (
          <div className="flex flex-col items-center justify-center h-full text-muted py-20">
            <div className="text-4xl mb-4 opacity-50">🏗️</div>
            <h2 className="text-xl font-semibold text-text">Page Under Construction</h2>
            <p className="mt-2 text-sm">Implementing the {activePage} module next...</p>
          </div>
        );
    }
  };

  return (
    <div className="flex min-h-screen bg-background text-text selection:bg-accent/30 selection:text-white font-body selection:bg-accent selection:text-nav overflow-hidden">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <Topbar title={getTitle()} />
        <div className="flex-1 overflow-y-auto p-7 scrollbar-hide">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
