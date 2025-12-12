import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Divider,
  Card,
  CardContent,
  Chip,
  CircularProgress,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { useSettingsStore } from '../store';
import { api } from '../services/api';

export default function Settings() {
  const { t, i18n } = useTranslation();
  const {
    language,
    temperature,
    topK,
    setLanguage,
    setTemperature,
    setTopK,
  } = useSettingsStore();

  // Fetch health status
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    refetchInterval: 10000,
  });

  // Fetch training stats
  const { data: trainingStats } = useQuery({
    queryKey: ['trainingStats'],
    queryFn: () => api.getTrainingStats(),
  });

  const handleLanguageChange = (newLanguage: 'fr' | 'en' | 'ar') => {
    setLanguage(newLanguage);
    i18n.changeLanguage(newLanguage);
    document.documentElement.dir = newLanguage === 'ar' ? 'rtl' : 'ltr';
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Typography variant="h5" sx={{ fontWeight: 600 }}>
        {t('settings.title')}
      </Typography>

      {/* Language & Model Settings */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Typography variant="h6" sx={{ mb: 3 }}>
          General
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Language Selection */}
          <FormControl fullWidth>
            <InputLabel>{t('settings.language')}</InputLabel>
            <Select
              value={language}
              label={t('settings.language')}
              onChange={(e) => handleLanguageChange(e.target.value as 'fr' | 'en' | 'ar')}
            >
              <MenuItem value="fr">🇫🇷 Français</MenuItem>
              <MenuItem value="en">🇬🇧 English</MenuItem>
              <MenuItem value="ar">🇸🇦 العربية</MenuItem>
            </Select>
          </FormControl>

          <Divider />

          {/* Temperature Slider */}
          <Box>
            <Typography gutterBottom>
              {t('settings.temperature')}: {temperature.toFixed(1)}
            </Typography>
            <Slider
              value={temperature}
              onChange={(_, value) => setTemperature(value as number)}
              min={0}
              max={1}
              step={0.1}
              marks={[
                { value: 0, label: '0' },
                { value: 0.5, label: '0.5' },
                { value: 1, label: '1' },
              ]}
            />
            <Typography variant="caption" color="text.secondary">
              Lower = more focused, Higher = more creative
            </Typography>
          </Box>

          {/* Top K Slider */}
          <Box>
            <Typography gutterBottom>
              {t('settings.topK')}: {topK}
            </Typography>
            <Slider
              value={topK}
              onChange={(_, value) => setTopK(value as number)}
              min={1}
              max={10}
              step={1}
              marks={[
                { value: 1, label: '1' },
                { value: 5, label: '5' },
                { value: 10, label: '10' },
              ]}
            />
            <Typography variant="caption" color="text.secondary">
              Number of relevant document chunks to retrieve
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* System Health */}
      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">
            {t('health.title')}
          </Typography>
          {health && (
            <Chip
              label={health.status === 'healthy' ? t('health.healthy') : 
                     health.status === 'degraded' ? t('health.degraded') : t('health.unhealthy')}
              color={getStatusColor(health.status) as 'success' | 'warning' | 'error' | 'default'}
            />
          )}
        </Box>

        {healthLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
            <CircularProgress size={32} />
          </Box>
        ) : (
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
            {health?.services.map((service) => (
              <Card key={service.name} variant="outlined">
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="subtitle2" sx={{ textTransform: 'capitalize' }}>
                      {service.name}
                    </Typography>
                    <Chip
                      label={service.status}
                      size="small"
                      color={getStatusColor(service.status) as 'success' | 'warning' | 'error' | 'default'}
                      variant="outlined"
                    />
                  </Box>
                  {service.latency_ms !== undefined && (
                    <Typography variant="caption" color="text.secondary">
                      Latency: {service.latency_ms}ms
                    </Typography>
                  )}
                </CardContent>
              </Card>
            ))}
          </Box>
        )}

        {health && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Last updated: {new Date(health.timestamp).toLocaleString()}
          </Typography>
        )}
      </Paper>

      {/* Training Stats */}
      {trainingStats && Object.keys(trainingStats).length > 0 && (
        <Paper
          elevation={0}
          sx={{
            p: 3,
            borderRadius: 3,
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="h6" sx={{ mb: 2 }}>
            Training Examples
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            {Object.entries(trainingStats).map(([lang, count]) => (
              <Chip
                key={lang}
                label={`${lang.toUpperCase()}: ${count}`}
                variant="outlined"
              />
            ))}
          </Box>
        </Paper>
      )}
    </Box>
  );
}
