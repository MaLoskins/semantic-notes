import { useState, useEffect } from 'react';
import apiService from '../services/api';

export default function ImportLocalNotesModal({ onClose, onImportComplete }) {
  const [localNotes, setLocalNotes] = useState([]);
  const [localTrash, setLocalTrash] = useState([]);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    // Check for localStorage notes
    const notesData = localStorage.getItem('semantic-notes-data');
    const trashData = localStorage.getItem('semantic-notes-trash');
    
    if (notesData) {
      try {
        const parsed = JSON.parse(notesData);
        setLocalNotes(parsed.notes || []);
      } catch (e) {
        console.error('Failed to parse notes:', e);
      }
    }
    
    if (trashData) {
      try {
        const parsed = JSON.parse(trashData);
        setLocalTrash(parsed.trash || []);
      } catch (e) {
        console.error('Failed to parse trash:', e);
      }
    }
  }, []);

  const handleImport = async () => {
    setImporting(true);
    setError('');
    
    try {
      const response = await apiService.request('/api/notes/import', {
        method: 'POST',
        body: JSON.stringify({
          notes: localNotes,
          trash: localTrash
        })
      });
      
      setSuccess(true);
      setTimeout(() => {
        onImportComplete(response.imported);
      }, 1500);
    } catch (err) {
      setError(err.message || 'Import failed. Please try again.');
      setImporting(false);
    }
  };

  const handleSkip = () => {
    onClose();
  };

  const totalToImport = localNotes.length + localTrash.length;

  if (totalToImport === 0) {
    return null; // Don't show modal if no notes to import
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content import-modal">
        <h2>Import Your Local Notes</h2>
        
        {!success ? (
          <>
            <p>
              We found <strong>{localNotes.length} notes</strong>
              {localTrash.length > 0 && ` and ${localTrash.length} items in trash`} 
              {' '}stored locally on this device.
            </p>
            
            <p>
              Would you like to import them to your account? This will:
            </p>
            
            <ul className="import-benefits">
              <li>✓ Sync your notes across devices</li>
              <li>✓ Keep your data safe in the cloud</li>
              <li>✓ Preserve all timestamps and tags</li>
              <li>✓ Maintain your local cache for speed</li>
            </ul>
            
            {error && (
              <div className="error-message">{error}</div>
            )}
            
            <div className="modal-actions">
              <button
                onClick={handleImport}
                disabled={importing}
                className="btn-primary"
              >
                {importing ? 'Importing...' : `Import ${totalToImport} Notes`}
              </button>
              <button
                onClick={handleSkip}
                disabled={importing}
                className="btn-secondary"
              >
                Skip for Now
              </button>
            </div>
            
            <p className="import-note">
              Note: Your local notes will remain available even if you skip.
            </p>
          </>
        ) : (
          <div className="success-message">
            <div className="success-icon">✓</div>
            <p>Successfully imported {totalToImport} notes!</p>
            <p>Redirecting...</p>
          </div>
        )}
      </div>
    </div>
  );
}