import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider, Typography, Chip } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';

export type View = 'chat' | 'documents' | 'settings';

interface SidebarProps {
  items: Array<{
    id: View;
    label: string;
    icon: React.ReactNode;
  }>;
  currentView: View;
  onViewChange: (view: View) => void;
}

export default function Sidebar({ items, currentView, onViewChange }: SidebarProps) {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.getHealth(),
    refetchInterval: 30000,
    retry: false,
  });

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
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <List sx={{ flexGrow: 1, px: 1 }}>
        {items.map((item) => (
          <ListItem key={item.id} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              selected={currentView === item.id}
              onClick={() => onViewChange(item.id)}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'primary.50',
                  color: 'primary.main',
                  '& .MuiListItemIcon-root': {
                    color: 'primary.main',
                  },
                },
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      {/* System Status */}
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          Status
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          {health?.services?.slice(0, 4).map((service: { name: string; status: string }) => (
            <Chip
              key={service.name}
              label={service.name}
              size="small"
              color={getStatusColor(service.status) as 'success' | 'warning' | 'error' | 'default'}
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
          ))}
        </Box>
        {health && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            v{health.version}
          </Typography>
        )}
      </Box>
    </Box>
  );
}
