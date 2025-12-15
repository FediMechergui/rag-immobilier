import { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Chip,
  Switch,
  FormControlLabel,
  Tooltip,
  Collapse,
  Card,
  CardContent,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  Send as SendIcon,
  Language as LanguageIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import { api, Source } from '../services/api';
import { useChatStore, useSettingsStore, Message } from '../store';

export default function Chat() {
  const { t } = useTranslation();
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const {
    messages,
    isLoading,
    webSearchEnabled,
    addMessage,
    updateMessage,
    appendToMessage,
    setLoading,
    setWebSearchEnabled,
  } = useChatStore();
  
  const { language, temperature, topK } = useSettingsStore();

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    
    // Add user message
    addMessage({
      role: 'user',
      content: userMessage,
    });

    // Add placeholder for assistant response
    const assistantId = addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true,
    });

    setLoading(true);

    try {
      // Use streaming API
      await api.queryStream(
        {
          question: userMessage,
          language,
          include_web_search: webSearchEnabled,
          top_k: topK,
          temperature,
        },
        // On chunk
        (chunk) => {
          appendToMessage(assistantId, chunk);
        },
        // On complete - don't overwrite content, just add metadata
        (response) => {
          updateMessage(assistantId, {
            sources: response.sources,
            queryId: response.query_id,
            isStreaming: false,
            webSearchUsed: response.web_search_used,
          });
        },
        // On error
        (error) => {
          updateMessage(assistantId, {
            content: `Error: ${error.message}`,
            isStreaming: false,
          });
        }
      );
    } catch (error) {
      updateMessage(assistantId, {
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        isStreaming: false,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (queryId: string, isHelpful: boolean) => {
    try {
      await api.submitFeedback({ query_id: queryId, is_helpful: isHelpful });
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const handleCopy = (content: string, id: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Messages Area */}
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          mb: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'text.secondary',
            }}
          >
            <Typography variant="h5" sx={{ mb: 1, fontWeight: 600 }}>
              🏠 {t('app.title')}
            </Typography>
            <Typography variant="body2" color="text.secondary" textAlign="center" maxWidth={400}>
              {t('app.subtitle')}
            </Typography>
          </Box>
        ) : (
          messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onFeedback={handleFeedback}
              onCopy={handleCopy}
              copiedId={copiedId}
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'background.paper',
        }}
      >
        {/* Web Search Toggle */}
        <Box sx={{ mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={webSearchEnabled}
                onChange={(e) => setWebSearchEnabled(e.target.checked)}
                size="small"
                color="secondary"
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <LanguageIcon fontSize="small" color={webSearchEnabled ? 'secondary' : 'disabled'} />
                <Typography variant="body2" color={webSearchEnabled ? 'secondary.main' : 'text.secondary'}>
                  {t('chat.webSearch')}
                </Typography>
              </Box>
            }
          />
        </Box>

        {/* Input Field */}
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder={t('chat.placeholder')}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            variant="outlined"
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
              },
            }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            sx={{
              width: 48,
              height: 48,
              backgroundColor: 'primary.main',
              color: 'white',
              '&:hover': {
                backgroundColor: 'primary.dark',
              },
              '&.Mui-disabled': {
                backgroundColor: 'action.disabledBackground',
                color: 'action.disabled',
              },
            }}
          >
            {isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
}

// Message Bubble Component
interface MessageBubbleProps {
  message: Message;
  onFeedback: (queryId: string, isHelpful: boolean) => void;
  onCopy: (content: string, id: string) => void;
  copiedId: string | null;
}

function MessageBubble({ message, onFeedback, onCopy, copiedId }: MessageBubbleProps) {
  const { t } = useTranslation();
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <Box
      className="message-animate"
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      <Paper
        elevation={0}
        sx={{
          maxWidth: "80%",
          p: 2,
          borderRadius: 2,
          backgroundColor: isUser ? "primary.main" : "background.paper",
          color: isUser ? "white" : "text.primary",
          border: isUser ? "none" : "1px solid",
          borderColor: "divider",
        }}
      >
        {/* Message Content */}
        <Box className="markdown-content">
          {message.isStreaming && !message.content ? (
            <Box sx={{ display: "flex", gap: 0.5 }}>
              <Box
                className="loading-dot"
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: "currentColor",
                }}
              />
              <Box
                className="loading-dot"
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: "currentColor",
                }}
              />
              <Box
                className="loading-dot"
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: "currentColor",
                }}
              />
            </Box>
          ) : (
            <>
              <ReactMarkdown>{message.content}</ReactMarkdown>
              {/* Bibliography at the end if sources exist */}
              {message.sources &&
                message.sources.length > 0 &&
                !message.isStreaming && (
                  <Box sx={{ mt: 2 }}>
                    <Typography
                      variant="subtitle2"
                      sx={{ fontWeight: 600, mb: 0.5 }}
                    >
                      {t("chat.sources") || "Ressources"}
                    </Typography>
                    <ol style={{ paddingLeft: 20, margin: 0 }}>
                      {message.sources.map((source, idx) => (
                        <li key={idx} style={{ marginBottom: 4 }}>
                          <span style={{ fontWeight: 500 }}>
                            {source.source_type === "web" ? "🌐 Web" : "📄 PDF"}
                          </span>{" "}
                          <span>
                            {source.document_name}
                            {source.page_number
                              ? `, page ${source.page_number}`
                              : ""}
                          </span>
                          {source.score !== undefined && (
                            <span
                              style={{
                                color: "#888",
                                marginLeft: 8,
                                fontSize: "0.85em",
                              }}
                            >
                              ({(source.score * 100).toFixed(0)}% pertinence)
                            </span>
                          )}
                        </li>
                      ))}
                    </ol>
                  </Box>
                )}
            </>
          )}
        </Box>

        {/* Web Search Badge */}
        {message.webSearchUsed && (
          <Chip
            icon={<LanguageIcon />}
            label={t("chat.webSearch")}
            size="small"
            color="secondary"
            variant="outlined"
            sx={{ mt: 1 }}
          />
        )}

        {/* Actions (for assistant messages) */}
        {!isUser && message.content && !message.isStreaming && (
          <Box
            sx={{
              mt: 1.5,
              display: "flex",
              alignItems: "center",
              gap: 1,
              flexWrap: "wrap",
            }}
          >
            {/* Copy Button */}
            <Tooltip title="Copy">
              <IconButton
                size="small"
                onClick={() => onCopy(message.content, message.id)}
                sx={{ color: "text.secondary" }}
              >
                {copiedId === message.id ? (
                  <CheckIcon fontSize="small" />
                ) : (
                  <CopyIcon fontSize="small" />
                )}
              </IconButton>
            </Tooltip>

            {/* Feedback Buttons */}
            {message.queryId && (
              <>
                <Tooltip title={t("chat.feedback.helpful")}>
                  <IconButton
                    size="small"
                    onClick={() => onFeedback(message.queryId!, true)}
                    sx={{ color: "text.secondary" }}
                  >
                    <ThumbUpIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title={t("chat.feedback.notHelpful")}>
                  <IconButton
                    size="small"
                    onClick={() => onFeedback(message.queryId!, false)}
                    sx={{ color: "text.secondary" }}
                  >
                    <ThumbDownIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </>
            )}

            {/* Sources Toggle */}
            {message.sources && message.sources.length > 0 && (
              <Button
                size="small"
                onClick={() => setShowSources(!showSources)}
                endIcon={showSources ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                sx={{ ml: "auto" }}
              >
                {t("chat.sources")} ({message.sources.length})
              </Button>
            )}
          </Box>
        )}

        {/* Sources Panel */}
        {message.sources && (
          <Collapse in={showSources}>
            <Box sx={{ mt: 2 }}>
              <SourcesPanel sources={message.sources} />
            </Box>
          </Collapse>
        )}
      </Paper>
    </Box>
  );
}

// Sources Panel Component
function SourcesPanel({ sources }: { sources: Source[] }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {sources.map((source, index) => (
        <Card key={index} variant="outlined" sx={{ backgroundColor: 'grey.50' }}>
          <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Chip
                label={source.source_type === 'web' ? '🌐 Web' : '📄 PDF'}
                size="small"
                variant="outlined"
              />
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                {source.document_name}
              </Typography>
              {source.page_number && (
                <Typography variant="caption" color="text.secondary">
                  • Page {source.page_number}
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                Score: {(source.score * 100).toFixed(0)}%
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.8rem' }}>
              {source.content.length > 200 ? source.content.substring(0, 200) + '...' : source.content}
            </Typography>
          </CardContent>
        </Card>
      ))}
    </Box>
  );
}
