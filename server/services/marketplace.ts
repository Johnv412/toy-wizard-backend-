export interface MarketplaceListing {
  id: string;
  title: string;
  description: string;
  price: number;
  imageUrl: string;
  url: string;
}

export interface IMarketplaceService {
  createListing(marketplace: string, listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing>;
  updateListing(marketplace: string, listingId: string, updates: Partial<MarketplaceListing>): Promise<MarketplaceListing>;
  deleteListing(marketplace: string, listingId: string): Promise<void>;
  getListingStatus(marketplace: string, listingId: string): Promise<string>;
}

export class MockMarketplaceService implements IMarketplaceService {
  async createListing(marketplace: string, listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockId = `mock_${marketplace}_${Date.now()}`;
    const mockUrl = `https://${marketplace}.com/item/${mockId}`;
    
    if (process.env.NODE_ENV !== "production") {
      console.log(`MOCK ${marketplace.toUpperCase()} LISTING CREATED:`, {
        id: mockId,
        title: listing.title,
        price: listing.price,
        url: mockUrl
      });
    }
    
    return {
      ...listing,
      id: mockId,
      url: mockUrl
    };
  }

  async updateListing(marketplace: string, listingId: string, updates: Partial<MarketplaceListing>): Promise<MarketplaceListing> {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (process.env.NODE_ENV !== "production") {
      console.log(`MOCK ${marketplace.toUpperCase()} LISTING UPDATED:`, {
        id: listingId,
        updates
      });
    }
    
    return {
      id: listingId,
      title: updates.title || 'Updated Listing',
      description: updates.description || 'Updated description',
      price: updates.price || 0,
      imageUrl: updates.imageUrl || '',
      url: `https://${marketplace}.com/item/${listingId}`
    };
  }

  async deleteListing(marketplace: string, listingId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 500));
    if (process.env.NODE_ENV !== "production") {
      console.log(`MOCK ${marketplace.toUpperCase()} LISTING DELETED:`, listingId);
    }
  }

  async getListingStatus(marketplace: string, listingId: string): Promise<string> {
    const statuses = ['active', 'sold', 'pending', 'cancelled'];
  return statuses[Math.floor(Math.random() * statuses.length)];
  }
}

export class LiveMarketplaceService implements IMarketplaceService {
  private ebayApiKey?: string;
  private amazonApiKey?: string;
  // Reminder: Ensure DB migrations/seeds are run in production if needed.
  
  constructor(ebayApiKey?: string, amazonApiKey?: string) {
    this.ebayApiKey = ebayApiKey;
    this.amazonApiKey = amazonApiKey;
  }

  async createListing(marketplace: string, listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    switch (marketplace.toLowerCase()) {
      case 'ebay':
        return this.createEbayListing(listing);
      case 'amazon':
        return this.createAmazonListing(listing);
      case 'facebook':
        return this.createFacebookListing(listing);
      case 'craigslist':
        return this.createCraigslistListing(listing);
      default:
        throw new Error(`Unsupported marketplace: ${marketplace}`);
    }
  }

  private async createEbayListing(listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    if (!this.ebayApiKey) {
      throw new Error('eBay API key not configured');
    }
    
    // Implementation would go here for actual eBay API
    throw new Error('eBay integration not yet implemented');
  }

  private async createAmazonListing(listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    if (!this.amazonApiKey) {
      throw new Error('Amazon API key not configured');
    }

    // Implementation would go here for actual Amazon API
    throw new Error('Amazon integration not yet implemented');
  }

  private async createFacebookListing(listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    // Facebook Marketplace API implementation would go here
    // For now, throw not implemented
    throw new Error('Facebook Marketplace integration not yet implemented');
  }

  private async createCraigslistListing(listing: Omit<MarketplaceListing, 'id' | 'url'>): Promise<MarketplaceListing> {
    // Craigslist API implementation would go here
    // For now, throw not implemented
    throw new Error('Craigslist integration not yet implemented');
  }

  async updateListing(marketplace: string, listingId: string, updates: Partial<MarketplaceListing>): Promise<MarketplaceListing> {
    throw new Error('Live marketplace update not yet implemented');
  }

  async deleteListing(marketplace: string, listingId: string): Promise<void> {
    throw new Error('Live marketplace delete not yet implemented');
  }

  async getListingStatus(marketplace: string, listingId: string): Promise<string> {
    throw new Error('Live marketplace status check not yet implemented');
  }
}

// Factory function
export function createMarketplaceService(): IMarketplaceService {
  const useMock = process.env.USE_MOCK_MARKETPLACE !== 'false';
  const ebayApiKey = process.env.EBAY_CLIENT_ID;
  const amazonApiKey = process.env.AMAZON_ACCESS_KEY;
  
  if (useMock || (!ebayApiKey && !amazonApiKey)) {
    console.log('Using mock marketplace service');
    return new MockMarketplaceService();
  }
  
  console.log('Using live marketplace service');
  return new LiveMarketplaceService(ebayApiKey, amazonApiKey);
}
