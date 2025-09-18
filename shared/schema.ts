import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, decimal, jsonb, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const toys = pgTable("toys", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  brand: text("brand"),
  category: text("category"),
  description: text("description"),
  condition: text("condition").notNull(), // 'mint', 'excellent', 'good', 'fair', 'poor'
  imageUrl: text("image_url").notNull(),
  aiAnalysis: jsonb("ai_analysis"), // AI analysis results
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const scans = pgTable("scans", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  toyId: varchar("toy_id").references(() => toys.id),
  imageUrl: text("image_url").notNull(),
  status: text("status").notNull().default('processing'), // 'processing', 'completed', 'failed'
  aiAnalysis: jsonb("ai_analysis"),
  estimatedPriceMin: decimal("estimated_price_min", { precision: 10, scale: 2 }),
  estimatedPriceMax: decimal("estimated_price_max", { precision: 10, scale: 2 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const listings = pgTable("listings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  scanId: varchar("scan_id").references(() => scans.id).notNull(),
  marketplace: text("marketplace").notNull(), // 'ebay', 'amazon', 'wordpress'
  marketplaceListingId: text("marketplace_listing_id"),
  title: text("title").notNull(),
  description: text("description"),
  price: decimal("price", { precision: 10, scale: 2 }).notNull(),
  status: text("status").notNull().default('draft'), // 'draft', 'active', 'sold', 'cancelled'
  listedAt: timestamp("listed_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const marketplaceConnections = pgTable("marketplace_connections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).notNull(),
  marketplace: text("marketplace").notNull(),
  isConnected: boolean("is_connected").default(false).notNull(),
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  expiresAt: timestamp("expires_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
});

export const insertToySchema = createInsertSchema(toys).omit({
  id: true,
  createdAt: true,
});

export const insertScanSchema = createInsertSchema(scans).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertListingSchema = createInsertSchema(listings).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertMarketplaceConnectionSchema = createInsertSchema(marketplaceConnections).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;

export type Toy = typeof toys.$inferSelect;
export type InsertToy = z.infer<typeof insertToySchema>;

export type Scan = typeof scans.$inferSelect;
export type InsertScan = z.infer<typeof insertScanSchema>;

export type Listing = typeof listings.$inferSelect;
export type InsertListing = z.infer<typeof insertListingSchema>;

export type MarketplaceConnection = typeof marketplaceConnections.$inferSelect;
export type InsertMarketplaceConnection = z.infer<typeof insertMarketplaceConnectionSchema>;
