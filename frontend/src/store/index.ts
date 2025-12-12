import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Source } from '../services/api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  queryId?: string;
  timestamp: Date;
  isStreaming?: boolean;
  webSearchUsed?: boolean;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  webSearchEnabled: boolean;
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  appendToMessage: (id: string, content: string) => void;
  setLoading: (loading: boolean) => void;
  setWebSearchEnabled: (enabled: boolean) => void;
  clearMessages: () => void;
}

interface SettingsState {
  language: 'fr' | 'en' | 'ar';
  theme: 'light' | 'dark';
  temperature: number;
  topK: number;
  setLanguage: (language: 'fr' | 'en' | 'ar') => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setTemperature: (temperature: number) => void;
  setTopK: (topK: number) => void;
}

interface DocumentState {
  refreshTrigger: number;
  triggerRefresh: () => void;
}

// Generate unique ID
const generateId = () => Math.random().toString(36).substring(2, 15);

// Chat store
export const useChatStore = create<ChatState>()((set) => ({
  messages: [],
  isLoading: false,
  webSearchEnabled: false,
  
  addMessage: (message) => {
    const id = generateId();
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id,
          timestamp: new Date(),
        },
      ],
    }));
    return id;
  },
  
  updateMessage: (id, updates) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      ),
    }));
  },
  
  appendToMessage: (id, content) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content: msg.content + content } : msg
      ),
    }));
  },
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setWebSearchEnabled: (enabled) => set({ webSearchEnabled: enabled }),
  
  clearMessages: () => set({ messages: [] }),
}));

// Settings store with persistence
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      language: 'fr',
      theme: 'light',
      temperature: 0.7,
      topK: 5,
      
      setLanguage: (language) => set({ language }),
      setTheme: (theme) => set({ theme }),
      setTemperature: (temperature) => set({ temperature }),
      setTopK: (topK) => set({ topK }),
    }),
    {
      name: 'rag-settings',
    }
  )
);

// Document store
export const useDocumentStore = create<DocumentState>()((set) => ({
  refreshTrigger: 0,
  triggerRefresh: () => set((state) => ({ refreshTrigger: state.refreshTrigger + 1 })),
}));
