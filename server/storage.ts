import { type User, type InsertUser, type Toy, type InsertToy, type Scan, type InsertScan, type Listing, type InsertListing, type MarketplaceConnection, type InsertMarketplaceConnection } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Users
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Toys
  getToy(id: string): Promise<Toy | undefined>;
  createToy(toy: InsertToy): Promise<Toy>;

  // Scans
  getScan(id: string): Promise<Scan | undefined>;
  getScansByUser(userId: string): Promise<Scan[]>;
  createScan(scan: InsertScan): Promise<Scan>;
  updateScan(id: string, updates: Partial<Scan>): Promise<Scan | undefined>;

  // Listings
  getListing(id: string): Promise<Listing | undefined>;
  getListingsByScan(scanId: string): Promise<Listing[]>;
  getListingsByUser(userId: string): Promise<Listing[]>;
  createListing(listing: InsertListing): Promise<Listing>;
  updateListing(id: string, updates: Partial<Listing>): Promise<Listing | undefined>;

  // Marketplace Connections
  getMarketplaceConnection(userId: string, marketplace: string): Promise<MarketplaceConnection | undefined>;
  getMarketplaceConnections(userId: string): Promise<MarketplaceConnection[]>;
  createMarketplaceConnection(connection: InsertMarketplaceConnection): Promise<MarketplaceConnection>;
  updateMarketplaceConnection(id: string, updates: Partial<MarketplaceConnection>): Promise<MarketplaceConnection | undefined>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User> = new Map();
  private toys: Map<string, Toy> = new Map();
  private scans: Map<string, Scan> = new Map();
  private listings: Map<string, Listing> = new Map();
  private marketplaceConnections: Map<string, MarketplaceConnection> = new Map();

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(user => user.username === username);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { 
      ...insertUser, 
      id, 
      createdAt: new Date() 
    };
    this.users.set(id, user);
    return user;
  }

  async getToy(id: string): Promise<Toy | undefined> {
    return this.toys.get(id);
  }

  async createToy(insertToy: InsertToy): Promise<Toy> {
    const id = randomUUID();
    const toy: Toy = { 
      ...insertToy, 
      id, 
      createdAt: new Date() 
    };
    this.toys.set(id, toy);
    return toy;
  }

  async getScan(id: string): Promise<Scan | undefined> {
    return this.scans.get(id);
  }

  async getScansByUser(userId: string): Promise<Scan[]> {
    return Array.from(this.scans.values())
      .filter(scan => scan.userId === userId)
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  async createScan(insertScan: InsertScan): Promise<Scan> {
    const id = randomUUID();
    const now = new Date();
    const scan: Scan = { 
      ...insertScan, 
      id, 
      createdAt: now,
      updatedAt: now
    };
    this.scans.set(id, scan);
    return scan;
  }

  async updateScan(id: string, updates: Partial<Scan>): Promise<Scan | undefined> {
    const scan = this.scans.get(id);
    if (!scan) return undefined;
    
    const updatedScan: Scan = { 
      ...scan, 
      ...updates, 
      updatedAt: new Date() 
    };
    this.scans.set(id, updatedScan);
    return updatedScan;
  }

  async getListing(id: string): Promise<Listing | undefined> {
    return this.listings.get(id);
  }

  async getListingsByScan(scanId: string): Promise<Listing[]> {
    return Array.from(this.listings.values()).filter(listing => listing.scanId === scanId);
  }

  async getListingsByUser(userId: string): Promise<Listing[]> {
    const userScans = await this.getScansByUser(userId);
    const scanIds = new Set(userScans.map(scan => scan.id));
    return Array.from(this.listings.values())
      .filter(listing => scanIds.has(listing.scanId))
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  async createListing(insertListing: InsertListing): Promise<Listing> {
    const id = randomUUID();
    const now = new Date();
    const listing: Listing = { 
      ...insertListing, 
      id, 
      createdAt: now,
      updatedAt: now
    };
    this.listings.set(id, listing);
    return listing;
  }

  async updateListing(id: string, updates: Partial<Listing>): Promise<Listing | undefined> {
    const listing = this.listings.get(id);
    if (!listing) return undefined;
    
    const updatedListing: Listing = { 
      ...listing, 
      ...updates, 
      updatedAt: new Date() 
    };
    this.listings.set(id, updatedListing);
    return updatedListing;
  }

  async getMarketplaceConnection(userId: string, marketplace: string): Promise<MarketplaceConnection | undefined> {
    return Array.from(this.marketplaceConnections.values())
      .find(conn => conn.userId === userId && conn.marketplace === marketplace);
  }

  async getMarketplaceConnections(userId: string): Promise<MarketplaceConnection[]> {
    return Array.from(this.marketplaceConnections.values())
      .filter(conn => conn.userId === userId);
  }

  async createMarketplaceConnection(insertConnection: InsertMarketplaceConnection): Promise<MarketplaceConnection> {
    const id = randomUUID();
    const now = new Date();
    const connection: MarketplaceConnection = { 
      ...insertConnection, 
      id, 
      createdAt: now,
      updatedAt: now
    };
    this.marketplaceConnections.set(id, connection);
    return connection;
  }

  async updateMarketplaceConnection(id: string, updates: Partial<MarketplaceConnection>): Promise<MarketplaceConnection | undefined> {
    const connection = this.marketplaceConnections.get(id);
    if (!connection) return undefined;
    
    const updatedConnection: MarketplaceConnection = { 
      ...connection, 
      ...updates, 
      updatedAt: new Date() 
    };
    this.marketplaceConnections.set(id, updatedConnection);
    return updatedConnection;
  }
}

export const storage = new MemStorage();
