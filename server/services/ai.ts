export interface AIAnalysisResult {
  toyName: string;
  brand?: string;
  category: string;
  condition: 'mint' | 'excellent' | 'good' | 'fair' | 'poor';
  description: string;
  estimatedPriceMin: number;
  estimatedPriceMax: number;
  confidence: number;
}

export interface IAIService {
  analyzeToy(imageUrl: string): Promise<AIAnalysisResult>;
}

export class MockAIService implements IAIService {
  async analyzeToy(imageUrl: string): Promise<AIAnalysisResult> {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Mock realistic toy analysis results
    const mockResults = [
      {
        toyName: "LEGO Creator Expert Big Ben",
        brand: "LEGO",
        category: "Building Sets",
        condition: "good" as const,
        description: "Large architectural building set with detailed Big Ben clock tower replica. Includes over 4,000 pieces.",
        estimatedPriceMin: 220,
        estimatedPriceMax: 280,
        confidence: 0.92
      },
      {
        toyName: "Vintage Barbie Doll",
        brand: "Mattel",
        category: "Dolls",
        condition: "excellent" as const,
        description: "Classic Barbie doll from the 1990s in original packaging. Blonde hair, pink dress.",
        estimatedPriceMin: 45,
        estimatedPriceMax: 85,
        confidence: 0.87
      },
      {
        toyName: "Hot Wheels Cars Set",
        brand: "Hot Wheels",
        category: "Die-cast Cars",
        condition: "good" as const,
        description: "Collection of vintage Hot Wheels cars from various years. Some wear on paint.",
        estimatedPriceMin: 15,
        estimatedPriceMax: 35,
        confidence: 0.78
      }
    ];
    
    return mockResults[Math.floor(Math.random() * mockResults.length)];
  }
}

export class OpenAIVisionService implements IAIService {
  private apiKey: string;
  
  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }
  
  async analyzeToy(imageUrl: string): Promise<AIAnalysisResult> {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4-vision-preview',
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'Analyze this toy image and provide: toy name, brand, category, condition (mint/excellent/good/fair/poor), description, and estimated resale price range in USD. Return as JSON with fields: toyName, brand, category, condition, description, estimatedPriceMin, estimatedPriceMax, confidence (0-1).'
              },
              {
                type: 'image_url',
                image_url: { url: imageUrl }
              }
            ]
          }
        ],
        max_tokens: 500
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content;
    
    try {
      return JSON.parse(content);
    } catch (error) {
      throw new Error('Failed to parse AI analysis response');
    }
  }
}

// Factory function
export function createAIService(): IAIService {
  const useMock = process.env.USE_MOCK_AI !== 'false';
  const openaiApiKey = process.env.OPENAI_API_KEY;
  
  if (useMock || !openaiApiKey) {
    console.log('Using mock AI service');
    return new MockAIService();
  }
  
  console.log('Using OpenAI Vision service');
  return new OpenAIVisionService(openaiApiKey);
}
