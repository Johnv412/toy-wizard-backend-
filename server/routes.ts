import type { Express } from "express";
import express from "express";
import { z } from "zod";
import { logger } from "./logger";
// --- Example validation for a user-facing route (adjust path as needed) ---
const listingSchema = z.object({
  title: z.string().min(1).max(140),
  price: z.number().nonnegative(),
  description: z.string().max(2000).optional(),
});

// Example: create a listing with validation
// (You can move this to the correct controller or route group as needed)
// POST /api/v1/listings
// Body: { title: string, price: number, description?: string }
// Returns: { ok: true } or { error: "INVALID_BODY", details }
//
// To use: Uncomment and connect to your DB logic
// app.post('/api/v1/listings', (req, res) => { ... })

// Example route:
// app.post('/api/v1/listings', (req, res) => {
//   const parse = listingSchema.safeParse(req.body);
//   if (!parse.success) {
//     return res.status(400).json({
//       error: "INVALID_BODY",
//       details: parse.error.flatten(),
//     });
//   }
//   const listing = parse.data;
//   // TODO: persist listing (DB call)
//   logger.info("new listing", { title: listing.title, price: listing.price });
//   return res.status(201).json({ ok: true });
// });
import { createServer, type Server } from "http";
import multer from "multer";
import path from "path";
import fs from "fs";
import { storage } from "./storage";
import { createAIService } from "./services/ai";
import { createMarketplaceService } from "./services/marketplace";
import { initializeWebSocketService, getWebSocketService } from "./services/websocket";
import { insertScanSchema, insertListingSchema, insertMarketplaceConnectionSchema } from "@shared/schema";

// Configure multer for file uploads
const uploadDir = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const upload = multer({
  dest: uploadDir,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // Initialize services
  const aiService = createAIService();
  const marketplaceService = createMarketplaceService();
  // const wsService = initializeWebSocketService(httpServer);

  // Kubernetes-style health checks
  app.get('/healthz', async (req, res) => {
    try {
      res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });
    } catch (error) {
      res.status(503).json({
        status: 'unhealthy',
        error: 'Service unavailable',
        timestamp: new Date().toISOString()
      });
    }
  });

  app.get('/readyz', async (req, res) => {
    try {
      // Test database connectivity
      await storage.getScan('demo-scan-id');
      
      res.status(200).json({
        status: 'ready',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        services: {
          database: 'connected',
          ai_service: process.env.USE_MOCK_AI !== 'false' ? 'mock' : 'configured',
          marketplace: process.env.USE_MOCK_MARKETPLACE !== 'false' ? 'mock' : 'configured'
        }
      });
    } catch (error) {
      res.status(503).json({
        status: 'not_ready',
        error: 'Dependencies not available',
        timestamp: new Date().toISOString(),
        services: {
          database: 'disconnected'
        }
      });
    }
  });

  // Detailed health check endpoint
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development',
      mock_mode: {
        ai_vision: process.env.USE_MOCK_AI !== 'false',
        marketplace: process.env.USE_MOCK_MARKETPLACE !== 'false',
        email: process.env.USE_MOCK_EMAIL !== 'false',
        payments: process.env.USE_MOCK_PAYMENTS !== 'false'
      },
      providers: {
        analysis: process.env.ANALYSIS_PROVIDER || 'mock',
        pricing: process.env.PRICING_PROVIDER || 'mock',
        email: process.env.EMAIL_PROVIDER || 'mock',
        payments: process.env.PAYMENTS_PROVIDER || 'mock'
      }
    });
  });

  // Scan endpoints
  app.post('/api/v1/scans/analyze', upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file provided' });
      }

      const userId = req.headers['x-user-id'] as string || 'demo-user';
      const imageUrl = `/uploads/${req.file.filename}`;

      // Create initial scan record
      const scan = await storage.createScan({
        userId,
        imageUrl,
        status: 'processing'
      });

      // Send immediate response with scan ID
      res.json({ scanId: scan.id, status: 'processing' });

      // Process AI analysis asynchronously
      try {
        // wsService.sendScanUpdate(userId, scan.id, 'analyzing');
        
        const analysis = await aiService.analyzeToy(imageUrl);
        
        // Update scan with results
        const updatedScan = await storage.updateScan(scan.id, {
          status: 'completed',
          aiAnalysis: analysis,
          estimatedPriceMin: analysis.estimatedPriceMin.toString(),
          estimatedPriceMax: analysis.estimatedPriceMax.toString()
        });

        // Create toy record
        await storage.createToy({
          name: analysis.toyName,
          brand: analysis.brand,
          category: analysis.category,
          condition: analysis.condition,
          description: analysis.description,
          imageUrl,
          aiAnalysis: analysis
        });

        // wsService.sendScanUpdate(userId, scan.id, 'completed', updatedScan);
      } catch (error) {
        console.error('AI analysis failed:', error);
        await storage.updateScan(scan.id, { status: 'failed' });
        // wsService.sendScanUpdate(userId, scan.id, 'failed', { error: error.message });
      }
    } catch (error) {
      console.error('Scan creation failed:', error);
      res.status(500).json({ error: 'Failed to create scan' });
    }
  });

  // FlipperZap alias endpoints (for generalized item analysis)
  app.post('/api/v1/analysis/analyze-item', upload.single('image'), async (req, res) => {
    // Reuse existing scan logic but with new endpoint name
    try {
      const userId = req.headers['x-user-id'] as string || 'demo-user';
      const imageUrl = req.file ? `/uploads/${req.file.filename}` : '';

      const scan = await storage.createScan({
        userId,
        imageUrl,
        status: 'pending'
      });

      res.json({ 
        analysisId: scan.id, 
        status: 'pending',
        message: 'Item analysis started'
      });

      // Process AI analysis asynchronously (same as original scan logic)
      try {
        // wsService.sendScanUpdate(userId, scan.id, 'analyzing');
        
        const analysis = await aiService.analyzeToy(imageUrl);
        
        const updatedScan = await storage.updateScan(scan.id, {
          status: 'completed',
          aiAnalysis: analysis,
          estimatedPriceMin: analysis.estimatedPriceMin.toString(),
          estimatedPriceMax: analysis.estimatedPriceMax.toString()
        });

        await storage.createToy({
          name: analysis.toyName,
          brand: analysis.brand,
          category: analysis.category,
          condition: analysis.condition,
          description: analysis.description,
          imageUrl,
          aiAnalysis: analysis
        });

        // wsService.sendScanUpdate(userId, scan.id, 'completed', updatedScan);
      } catch (error) {
        console.error('AI analysis failed:', error);
        await storage.updateScan(scan.id, { status: 'failed' });
        // wsService.sendScanUpdate(userId, scan.id, 'failed', { error: error.message });
      }
    } catch (error) {
      console.error('Item analysis failed:', error);
      res.status(500).json({ error: 'Failed to analyze item' });
    }
  });

  // Pricing history alias
  app.get('/api/v1/pricing/history/:item_name', async (req, res) => {
    try {
      const itemName = req.params.item_name;
      // Mock pricing history for demo mode
      const history = [
        { date: '2025-01-01', price: 45.99, condition: 8, marketplace: 'eBay' },
        { date: '2024-12-15', price: 42.50, condition: 7, marketplace: 'Mercari' },
        { date: '2024-12-01', price: 38.00, condition: 6, marketplace: 'Facebook' }
      ];
      res.json(history);
    } catch (error) {
      console.error('Failed to fetch pricing history:', error);
      res.status(500).json({ error: 'Failed to fetch pricing history' });
    }
  });

  // Pricing estimate alias  
  app.get('/api/v1/pricing/estimate', async (req, res) => {
    try {
      const { item_name, condition } = req.query;
      const basePrice = 40; // Mock base price
      const conditionMultiplier = (parseInt(condition as string) || 7) / 10;
      const low = Math.round(basePrice * conditionMultiplier * 0.8);
      const high = Math.round(basePrice * conditionMultiplier * 1.3);
      
      res.json({
        low,
        high,
        confidence: 0.82,
        item_name,
        condition: parseInt(condition as string)
      });
    } catch (error) {
      console.error('Failed to get pricing estimate:', error);
      res.status(500).json({ error: 'Failed to get pricing estimate' });
    }
  });

  app.get('/api/v1/scans', async (req, res) => {
    try {
      const userId = req.headers['x-user-id'] as string || 'demo-user';
      const scans = await storage.getScansByUser(userId);
      res.json(scans);
    } catch (error) {
      console.error('Failed to fetch scans:', error);
      res.status(500).json({ error: 'Failed to fetch scans' });
    }
  });

  app.get('/api/v1/scans/:id', async (req, res) => {
    try {
      const scan = await storage.getScan(req.params.id);
      if (!scan) {
        return res.status(404).json({ error: 'Scan not found' });
      }
      res.json(scan);
    } catch (error) {
      console.error('Failed to fetch scan:', error);
      res.status(500).json({ error: 'Failed to fetch scan' });
    }
  });

  // Listing endpoints
  app.post('/api/v1/listings', async (req, res) => {
    try {
      const listingData = insertListingSchema.parse(req.body);
      const listing = await storage.createListing(listingData);

      // If auto-list is enabled, create marketplace listing
      if (req.body.autoList && req.body.marketplace) {
        try {
          const marketplaceListing = await marketplaceService.createListing(
            req.body.marketplace,
            {
              title: listing.title,
              description: listing.description || '',
              price: parseFloat(listing.price),
              imageUrl: req.body.imageUrl || ''
            }
          );

          await storage.updateListing(listing.id, {
            marketplaceListingId: marketplaceListing.id,
            status: 'active',
            listedAt: new Date()
          });

          const userId = req.headers['x-user-id'] as string || 'demo-user';
          getWebSocketService().sendListingUpdate(userId, listing.id, 'listed', marketplaceListing);
        } catch (error) {
          console.error('Marketplace listing failed:', error);
        }
      }

      res.json(listing);
    } catch (error) {
      console.error('Failed to create listing:', error);
      res.status(500).json({ error: 'Failed to create listing' });
    }
  });

  app.get('/api/v1/listings', async (req, res) => {
    try {
      const userId = req.headers['x-user-id'] as string || 'demo-user';
      const listings = await storage.getListingsByUser(userId);
      res.json(listings);
    } catch (error) {
      console.error('Failed to fetch listings:', error);
      res.status(500).json({ error: 'Failed to fetch listings' });
    }
  });

  app.get('/api/v1/listings/scan/:scanId', async (req, res) => {
    try {
      const listings = await storage.getListingsByScan(req.params.scanId);
      res.json(listings);
    } catch (error) {
      console.error('Failed to fetch listings:', error);
      res.status(500).json({ error: 'Failed to fetch listings' });
    }
  });

  // Marketplace connection endpoints
  app.get('/api/v1/marketplace/connections', async (req, res) => {
    try {
      const userId = req.headers['x-user-id'] as string || 'demo-user';
      const connections = await storage.getMarketplaceConnections(userId);
      
      // Return status for all supported marketplaces
      const marketplaces = ['ebay', 'amazon', 'facebook', 'craigslist', 'wordpress'];
      const status = marketplaces.map(marketplace => {
        const connection = connections.find(c => c.marketplace === marketplace);
        return {
          marketplace,
          connected: connection?.isConnected || false,
          lastUpdated: connection?.updatedAt
        };
      });
      
      res.json(status);
    } catch (error) {
      console.error('Failed to fetch marketplace connections:', error);
      res.status(500).json({ error: 'Failed to fetch marketplace connections' });
    }
  });

  app.post('/api/v1/marketplace/connect', async (req, res) => {
    try {
      const connectionData = insertMarketplaceConnectionSchema.parse(req.body);
      const userId = req.headers['x-user-id'] as string || 'demo-user';
      
      const existingConnection = await storage.getMarketplaceConnection(userId, connectionData.marketplace);
      
      if (existingConnection) {
        const updated = await storage.updateMarketplaceConnection(existingConnection.id, {
          isConnected: true,
          ...connectionData
        });
        res.json(updated);
      } else {
        const connection = await storage.createMarketplaceConnection({
          ...connectionData,
          userId,
          isConnected: true
        });
        res.json(connection);
      }
    } catch (error) {
      console.error('Failed to connect marketplace:', error);
      res.status(500).json({ error: 'Failed to connect marketplace' });
    }
  });

  // Serve uploaded images
  app.use('/uploads', express.static(uploadDir));

  return httpServer;
}