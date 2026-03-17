
import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { initializeApp, getApps } from 'firebase-admin/app';
import firebaseConfig from './firebase-applet-config.json' assert { type: 'json' };

async function startServer() {
  const app = express();
  app.use(express.json());

  // Initialize Firebase Admin
  if (!getApps().length) {
    initializeApp({
      projectId: firebaseConfig.projectId,
    });
  }

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(3000, '0.0.0.0', () => {
    console.log('Server running on http://localhost:3000');
  });
}

startServer();
