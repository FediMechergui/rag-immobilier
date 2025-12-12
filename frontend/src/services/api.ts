import axios, { AxiosInstance, AxiosError } from 'axios';

// Types
export interface Document {
  id: string;
  filename: string;
  status: 'pending' | 'processing' | 'ready' | 'error';
  chunk_count: number;
  created_at: string;
  error_message?: string;
}

export interface Source {
  document_id: string;
  document_name: string;
  chunk_index: number;
  content: string;
  score: number;
  page_number?: number;
  source_type: 'pdf' | 'web';
}

export interface QueryRequest {
  question: string;
  language?: 'fr' | 'en' | 'ar';
  include_web_search?: boolean;
  top_k?: number;
  temperature?: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  query_id: string;
  processing_time_ms: number;
  web_search_used: boolean;
}

export interface FeedbackRequest {
  query_id: string;
  is_helpful: boolean;
  comment?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: ServiceHealth[];
  version: string;
}

export interface ServiceHealth {
  name: string;
  status: string;
  latency_ms?: number;
  details?: Record<string, unknown>;
}

export interface TrainingExample {
  id: string;
  question: string;
  answer: string;
  language: string;
  created_at: string;
  is_active: boolean;
}

export interface TrainingExampleCreate {
  question: string;
  answer: string;
  language?: string;
}

// API Client
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 300000, // 5 minutes for long operations
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message = (error.response?.data as { detail?: string })?.detail || error.message;
        console.error('API Error:', message);
        throw new Error(message);
      }
    );
  }

  // Health
  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health', {
      baseURL: '', // Health endpoint is at root, not under /api
    });
    return response.data;
  }

  // Documents
  async uploadDocument(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<Document>('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async getDocuments(): Promise<Document[]> {
    const response = await this.client.get<Document[]>('/documents/');
    return response.data;
  }

  async getDocument(id: string): Promise<Document> {
    const response = await this.client.get<Document>(`/documents/${id}`);
    return response.data;
  }

  async deleteDocument(id: string): Promise<void> {
    await this.client.delete(`/documents/${id}`);
  }

  async getDocumentStatus(id: string): Promise<{ status: string; chunk_count: number }> {
    const response = await this.client.get(`/documents/${id}/status`);
    return response.data;
  }

  // Query
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await this.client.post<QueryResponse>('/query/', request);
    return response.data;
  }

  async queryStream(
    request: QueryRequest,
    onChunk: (chunk: string) => void,
    onComplete: (response: QueryResponse) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let finalResponse: QueryResponse | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              continue;
            }
            
            try {
              const parsed = JSON.parse(data);
              // Handle backend format: { chunk, done, sources?, processing_time_ms?, error? }
              if (parsed.error) {
                throw new Error(parsed.error);
              }
              if (parsed.chunk) {
                onChunk(parsed.chunk);
              }
              if (parsed.done && parsed.sources) {
                // Transform backend sources to frontend format
                const mappedSources: Source[] = parsed.sources.map((s: any) => ({
                  document_id: s.document_id || '',
                  document_name: s.filename || s.title || 'Unknown',
                  chunk_index: s.chunk_index || 0,
                  content: s.chunk_preview || '',
                  score: s.relevance_score || 0,
                  page_number: s.page || undefined,
                  source_type: s.type === 'web' ? 'web' : 'pdf',
                }));
                
                finalResponse = {
                  answer: '', // Will be built from chunks
                  sources: mappedSources,
                  query_id: '',
                  processing_time_ms: parsed.processing_time_ms || 0,
                  web_search_used: false,
                };
              }
            } catch (e) {
              // Not JSON or parse error
              if (e instanceof SyntaxError) {
                onChunk(data);
              } else {
                throw e;
              }
            }
          }
        }
      }

      if (finalResponse) {
        onComplete(finalResponse);
      }
    } catch (error) {
      onError(error instanceof Error ? error : new Error(String(error)));
    }
  }

  async submitFeedback(feedback: FeedbackRequest): Promise<void> {
    await this.client.post('/query/feedback', feedback);
  }

  // Training
  async getTrainingExamples(language?: string): Promise<TrainingExample[]> {
    const params = language ? { language } : {};
    const response = await this.client.get<TrainingExample[]>('/training/', { params });
    return response.data;
  }

  async createTrainingExample(example: TrainingExampleCreate): Promise<TrainingExample> {
    const response = await this.client.post<TrainingExample>('/training/', example);
    return response.data;
  }

  async updateTrainingExample(
    id: string,
    example: Partial<TrainingExampleCreate & { is_active: boolean }>
  ): Promise<TrainingExample> {
    const response = await this.client.put<TrainingExample>(`/training/${id}`, example);
    return response.data;
  }

  async deleteTrainingExample(id: string): Promise<void> {
    await this.client.delete(`/training/${id}`);
  }

  async getTrainingStats(): Promise<Record<string, number>> {
    const response = await this.client.get('/training/stats');
    return response.data;
  }
}

export const api = new ApiClient();
