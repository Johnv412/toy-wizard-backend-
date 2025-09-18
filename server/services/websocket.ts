import { WebSocket, WebSocketServer } from 'ws';
import { Server } from 'http';

export interface WebSocketMessage {
  type: string;
  payload: any;
  userId?: string;
}

export class WebSocketService {
  private wss: WebSocketServer;
  private clients: Map<string, Set<WebSocket>> = new Map();

  constructor(server: Server) {
    this.wss = new WebSocketServer({ 
      server,
      path: '/ws/api' // Use a specific path to avoid conflicts with Vite's WebSocket
    });
    this.setupWebSocketServer();
  }

  private setupWebSocketServer() {
    this.wss.on('connection', (ws: WebSocket, request) => {
      console.log('WebSocket client connected');
      
      ws.on('message', (data: Buffer) => {
        try {
          const message: WebSocketMessage = JSON.parse(data.toString());
          this.handleMessage(ws, message);
        } catch (error) {
          console.error('Invalid WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        console.log('WebSocket client disconnected');
        this.removeClient(ws);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      // Send welcome message
      this.sendMessage(ws, {
        type: 'connection',
        payload: { status: 'connected', timestamp: new Date().toISOString() }
      });
    });
  }

  private handleMessage(ws: WebSocket, message: WebSocketMessage) {
    switch (message.type) {
      case 'register':
        if (message.userId) {
          this.registerClient(message.userId, ws);
        }
        break;
      case 'ping':
        this.sendMessage(ws, { type: 'pong', payload: {} });
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  }

  private registerClient(userId: string, ws: WebSocket) {
    if (!this.clients.has(userId)) {
      this.clients.set(userId, new Set());
    }
    this.clients.get(userId)!.add(ws);
    console.log(`Client registered for user: ${userId}`);
  }

  private removeClient(ws: WebSocket) {
    this.clients.forEach((clients, userId) => {
      if (clients.has(ws)) {
        clients.delete(ws);
        if (clients.size === 0) {
          this.clients.delete(userId);
        }
        console.log(`Client removed for user: ${userId}`);
      }
    });
  }

  private sendMessage(ws: WebSocket, message: WebSocketMessage) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  // Public methods for sending notifications
  public sendToUser(userId: string, message: WebSocketMessage) {
    const clients = this.clients.get(userId);
    if (clients) {
      clients.forEach(ws => {
        this.sendMessage(ws, message);
      });
    }
  }

  public broadcastToAll(message: WebSocketMessage) {
    this.wss.clients.forEach(ws => {
      this.sendMessage(ws, message);
    });
  }

  public sendScanUpdate(userId: string, scanId: string, status: string, data?: any) {
    this.sendToUser(userId, {
      type: 'scan_update',
      payload: {
        scanId,
        status,
        data,
        timestamp: new Date().toISOString()
      }
    });
  }

  public sendListingUpdate(userId: string, listingId: string, status: string, data?: any) {
    this.sendToUser(userId, {
      type: 'listing_update',
      payload: {
        listingId,
        status,
        data,
        timestamp: new Date().toISOString()
      }
    });
  }
}

let wsService: WebSocketService | null = null;

export function initializeWebSocketService(server: Server): WebSocketService {
  wsService = new WebSocketService(server);
  return wsService;
}

export function getWebSocketService(): WebSocketService {
  if (!wsService) {
    throw new Error('WebSocket service not initialized');
  }
  return wsService;
}
