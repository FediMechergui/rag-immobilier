import { useState, useEffect } from 'react';
import { Box, Container, AppBar, Toolbar, Typography, IconButton, Drawer, useMediaQuery, useTheme as useMuiTheme, Theme } from '@mui/material';
import { Menu as MenuIcon, Description as DescriptionIcon, Settings as SettingsIcon, Chat as ChatIcon } from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import Chat from './components/Chat';
import Documents from './components/Documents';
import Settings from './components/Settings';
import Sidebar, { type View } from './components/Sidebar';
import { useSettingsStore } from './store';

function App() {
  const { t, i18n } = useTranslation();
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [currentView, setCurrentView] = useState<View>('chat');
  const { language } = useSettingsStore();

  useEffect(() => {
    i18n.changeLanguage(language);
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.lang = language;
  }, [language, i18n]);

  const handleViewChange = (view: View) => {
    setCurrentView(view);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  const renderContent = () => {
    switch (currentView) {
      case 'chat':
        return <Chat />;
      case 'documents':
        return <Documents />;
      case 'settings':
        return <Settings />;
      default:
        return <Chat />;
    }
  };

  const navigationItems = [
    { id: 'chat' as View, label: 'Chat', icon: <ChatIcon /> },
    { id: 'documents' as View, label: t('documents.title'), icon: <DescriptionIcon /> },
    { id: 'settings' as View, label: t('settings.title'), icon: <SettingsIcon /> },
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme: Theme) => theme.zIndex.drawer + 1,
          backgroundColor: 'white',
          color: 'text.primary',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        }}
      >
        <Toolbar>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={() => setDrawerOpen(!drawerOpen)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box
              sx={{
                width: 36,
                height: 36,
                borderRadius: 1.5,
                background: 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontWeight: 700,
                fontSize: '1.1rem',
              }}
            >
              🏠
            </Box>
            <Box>
              <Typography variant="h6" component="h1" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
                {t('app.title')}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: { xs: 'none', sm: 'block' } }}>
                {t('app.subtitle')}
              </Typography>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Sidebar / Drawer */}
      <Drawer
        variant={isMobile ? 'temporary' : 'permanent'}
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sx={{
          width: 240,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 240,
            boxSizing: 'border-box',
            borderRight: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'background.paper',
          },
        }}
      >
        <Toolbar />
        <Sidebar
          items={navigationItems}
          currentView={currentView}
          onViewChange={handleViewChange}
        />
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 0,
          width: { md: `calc(100% - 240px)` },
          ml: { md: 0 },
          backgroundColor: 'background.default',
        }}
      >
        <Toolbar />
        <Container 
          maxWidth="lg" 
          sx={{ 
            py: 3, 
            height: 'calc(100vh - 64px)',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {renderContent()}
        </Container>
      </Box>
    </Box>
  );
}

export default App;
