import { useCallback, useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Description as DescriptionIcon,
  CloudUpload as CloudUploadIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as ProcessingIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { api, Document } from '../services/api';

export default function Documents() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<Document | null>(null);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
    open: false,
    message: '',
    severity: 'success',
  });

  // Fetch documents
  const { data: documents = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(),
    refetchInterval: 5000, // Refresh every 5 seconds to update processing status
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setSnackbar({ open: true, message: t('documents.uploadSuccess'), severity: 'success' });
    },
    onError: (error: Error) => {
      setSnackbar({ open: true, message: `${t('documents.uploadError')}: ${error.message}`, severity: 'error' });
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setDeleteDialogOpen(false);
      setDocumentToDelete(null);
    },
  });

  // Dropzone
  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      uploadMutation.mutate(file);
    });
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleDeleteClick = (document: Document) => {
    setDocumentToDelete(document);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (documentToDelete) {
      deleteMutation.mutate(documentToDelete.id);
    }
  };

  const getStatusIcon = (status: Document['status']) => {
    switch (status) {
      case 'ready':
        return <CheckCircleIcon color="success" />;
      case 'processing':
        return <ProcessingIcon color="primary" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <ProcessingIcon color="disabled" />;
    }
  };

  const getStatusLabel = (status: Document['status']) => {
    switch (status) {
      case 'ready':
        return t('documents.ready');
      case 'processing':
        return t('documents.processing');
      case 'error':
        return t('documents.error');
      default:
        return 'Pending';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Typography variant="h5" sx={{ fontWeight: 600 }}>
        {t('documents.title')}
      </Typography>

      {/* Upload Zone */}
      <Paper
        {...getRootProps()}
        elevation={0}
        sx={{
          p: 4,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'divider',
          borderRadius: 3,
          backgroundColor: isDragActive ? 'primary.50' : 'background.paper',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'primary.50',
          },
        }}
      >
        <input {...getInputProps()} />
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
          <CloudUploadIcon sx={{ fontSize: 48, color: isDragActive ? 'primary.main' : 'text.secondary' }} />
          <Typography variant="h6" color={isDragActive ? 'primary.main' : 'text.primary'}>
            {t('documents.upload')}
          </Typography>
          <Typography variant="body2" color="text.secondary" textAlign="center">
            {t('documents.uploadHint')}
          </Typography>
        </Box>
      </Paper>

      {/* Upload Progress */}
      {uploadMutation.isPending && (
        <LinearProgress sx={{ borderRadius: 1 }} />
      )}

      {/* Documents List */}
      <Paper
        elevation={0}
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          borderRadius: 3,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        {isLoading ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography color="text.secondary">{t('common.loading')}</Typography>
          </Box>
        ) : documents.length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <DescriptionIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
            <Typography color="text.secondary">{t('documents.noDocuments')}</Typography>
          </Box>
        ) : (
          <List sx={{ py: 0 }}>
            {documents.map((doc, index) => (
              <ListItem
                key={doc.id}
                divider={index < documents.length - 1}
                sx={{ py: 2 }}
              >
                <Box sx={{ mr: 2 }}>
                  {getStatusIcon(doc.status)}
                </Box>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {doc.filename}
                      </Typography>
                      <Chip
                        label={getStatusLabel(doc.status)}
                        size="small"
                        color={
                          doc.status === 'ready' ? 'success' :
                          doc.status === 'processing' ? 'primary' :
                          doc.status === 'error' ? 'error' : 'default'
                        }
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(doc.created_at)}
                      </Typography>
                      {doc.chunk_count > 0 && (
                        <Typography variant="caption" color="text.secondary">
                          {doc.chunk_count} {t('documents.chunks')}
                        </Typography>
                      )}
                      {doc.error_message && (
                        <Typography variant="caption" color="error">
                          {doc.error_message}
                        </Typography>
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <IconButton
                    edge="end"
                    onClick={() => handleDeleteClick(doc)}
                    disabled={deleteMutation.isPending}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>{t('documents.delete')}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {t('documents.deleteConfirm')}
            {documentToDelete && (
              <Box component="span" sx={{ fontWeight: 500, display: 'block', mt: 1 }}>
                {documentToDelete.filename}
              </Box>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>{t('common.cancel')}</Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            disabled={deleteMutation.isPending}
          >
            {t('common.delete')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
