import React, { useState, useEffect } from 'react';
import {
  Theme,
  Content,
  Grid,
  Column,
  Tile,
  Button,
  TextInput,
  Tag,
  Loading,
  InlineNotification,
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  ProgressIndicator,
  ProgressStep,
  ProgressBar,
  InlineLoading,
} from '@carbon/react';
import {
  Search,
  Compare,
  Renew,
  CloudUpload,
  CheckmarkFilled,
  ErrorFilled,
  Chat,
  FlowData,
} from '@carbon/icons-react';

import {
  queryStandardRAG,
  queryWorkflowAgent,
  getDataStatus,
  populateData,
  getPopulateStatus,
  getMetrics,
  resetSession,
} from './services/api';

// Helper function to render markdown-style formatting
function renderMarkdown(text) {
  if (!text) return '';
  
  // Convert **bold** to <strong>
  let formatted = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  
  // Convert *italic* to <em>
  formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
  
  // Convert `code` to <code>
  formatted = formatted.replace(/`(.+?)`/g, '<code style="background: #393939; padding: 0.125rem 0.25rem; border-radius: 2px;">$1</code>');
  
  return formatted;
}

function App() {
  // Data population state
  const [dataStatus, setDataStatus] = useState(null);
  const [dataStatusLoading, setDataStatusLoading] = useState(true);
  const [populateState, setPopulateState] = useState({ status: 'idle' });

  // Query state
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  // Results
  const [standardResult, setStandardResult] = useState(null);
  const [workflowResult, setWorkflowResult] = useState(null);
  const [metrics, setMetrics] = useState(null);

  // Active tab
  const [selectedTab, setSelectedTab] = useState(0);

  // ── Check data status on mount ──
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await getDataStatus();
        setDataStatus(status);
      } catch (err) {
        setDataStatus({ populated: false, document_count: 0, message: 'Backend not reachable' });
      } finally {
        setDataStatusLoading(false);
      }
    };
    checkStatus();
  }, []);

  // ── Populate data handler ──
  const handlePopulate = async () => {
    try {
      setPopulateState({ status: 'running', progress: 0 });
      await populateData();
      
      let pollCount = 0;
      const MAX_POLLS = 300; // 10 minutes (300 * 2 seconds)
      
      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          pollCount++;
          
          if (pollCount > MAX_POLLS) {
            clearInterval(pollInterval);
            setPopulateState({
              status: 'error',
              error: 'Population timeout - check backend logs for details'
            });
            return;
          }
          
          const status = await getPopulateStatus();
          console.log('Populate status:', status); // Log for debugging
          setPopulateState(status);
          
          if (status.status === 'done') {
            clearInterval(pollInterval);
            setTimeout(async () => {
              const newStatus = await getDataStatus();
              setDataStatus(newStatus);
            }, 1500);
          } else if (status.status === 'error') {
            clearInterval(pollInterval);
          }
        } catch (err) {
          console.error('Polling error:', err); // Log errors
          // Continue polling on network errors
        }
      }, 2000);
    } catch (err) {
      console.error('Populate error:', err);
      setPopulateState({ status: 'error', error: err.message });
    }
  };

  // ── Query handler ──
  const handleQuery = async (type) => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (type === 'standard') {
        const result = await queryStandardRAG(question, 5);
        setStandardResult(result);
        setSelectedTab(0);
      } else if (type === 'workflow') {
        const result = await queryWorkflowAgent(question, sessionId, 5);
        setWorkflowResult(result);
        if (!sessionId) setSessionId(result.session_id);
        setSelectedTab(1);
      } else if (type === 'both') {
        const [standardRes, workflowRes] = await Promise.all([
          queryStandardRAG(question, 5),
          queryWorkflowAgent(question, sessionId, 5),
        ]);
        setStandardResult(standardRes);
        setWorkflowResult(workflowRes);
        if (!sessionId) setSessionId(workflowRes.session_id);
        setSelectedTab(2);
      }

      const metricsData = await getMetrics();
      setMetrics(metricsData);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  // ── Reset session handler ──
  const handleResetSession = async () => {
    if (sessionId) {
      try {
        await resetSession(sessionId);
        setSessionId(null);
        setWorkflowResult(null);
      } catch (err) {
        console.error('Failed to reset session:', err);
      }
    }
  };

  return (
    <Theme theme="g100">
      <Content style={{ background: '#161616', minHeight: '100vh', padding: '2rem' }}>
        <Grid>
          <Column lg={16} md={8} sm={4}>
            {/* Header */}
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                <FlowData size={32} style={{ color: '#0f62fe' }} />
                <h1 style={{ margin: 0 }}>Multiturn Workflow Agent Demo</h1>
              </div>
              <p style={{ color: '#c6c6c6' }}>
                Compare Standard RAG (vector-only) vs Workflow Agent (conversational + orchestration) for operational workflows
              </p>
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
                <Tag type="blue">IBM watsonx</Tag>
                <Tag type="purple">OpenSearch</Tag>
                <Tag type="teal">Conversational AI</Tag>
                {dataStatus?.document_count > 0 && (
                  <Tag type="cool-gray">{dataStatus.document_count} Workflows</Tag>
                )}
              </div>
            </div>

            {/* Loading spinner */}
            {dataStatusLoading && (
              <Tile style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <Loading description="Checking data status..." withOverlay={false} small />
                <p style={{ color: '#8d8d8d', margin: 0 }}>Connecting to OpenSearch...</p>
              </Tile>
            )}

            {/* Populate panel */}
            {!dataStatusLoading && !dataStatus?.populated && (
              <Tile style={{ marginBottom: '2rem', borderLeft: '4px solid #0f62fe' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <CloudUpload size={24} style={{ color: '#0f62fe' }} />
                  <h4 style={{ margin: 0 }}>No Data in OpenSearch</h4>
                  <Tag type="red" size="sm" style={{ marginLeft: 'auto' }}>Index Empty</Tag>
                </div>

                {populateState.status === 'idle' && (
                  <>
                    <p style={{ color: '#c6c6c6', marginBottom: '1rem' }}>
                      Generate and ingest 1,000 synthetic workflow documents with IBM watsonx embeddings.
                    </p>
                    <Button renderIcon={CloudUpload} onClick={handlePopulate}>
                      Populate Data
                    </Button>
                  </>
                )}

                {populateState.status === 'running' && (
                  <>
                    <InlineLoading description={populateState.message || 'Processing...'} status="active" />
                    <ProgressBar
                      value={populateState.progress || 0}
                      max={100}
                      label={`${populateState.progress || 0}% complete`}
                      style={{ marginTop: '1rem' }}
                    />
                  </>
                )}

                {populateState.status === 'done' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <CheckmarkFilled size={24} style={{ color: '#42be65' }} />
                    <p style={{ color: '#42be65', margin: 0 }}>Data populated successfully!</p>
                  </div>
                )}

                {populateState.status === 'error' && (
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
                    <ErrorFilled size={24} style={{ color: '#fa4d56', marginTop: '0.25rem' }} />
                    <div style={{ flex: 1 }}>
                      <p style={{ color: '#fa4d56', margin: 0, fontWeight: 600 }}>Population failed</p>
                      {populateState.error && (
                        <p style={{ color: '#c6c6c6', fontSize: '0.875rem', marginTop: '0.5rem', marginBottom: '0.5rem' }}>
                          <strong>Error:</strong> {populateState.error}
                        </p>
                      )}
                      {populateState.message && populateState.message !== `Population failed: ${populateState.error}` && (
                        <p style={{ color: '#c6c6c6', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
                          {populateState.message}
                        </p>
                      )}
                      <Button kind="secondary" size="sm" onClick={() => setPopulateState({ status: 'idle' })} style={{ marginTop: '0.5rem' }}>
                        Try Again
                      </Button>
                    </div>
                  </div>
                )}
              </Tile>
            )}

            {/* Query interface */}
            {dataStatus?.populated && (
              <>
                <Tile style={{ marginBottom: '2rem' }}>
                  <TextInput
                    id="question-input"
                    labelText="Ask a question or give an instruction"
                    placeholder="e.g., Start the high CPU troubleshooting workflow for server api-prod-01"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !loading) handleQuery('both');
                    }}
                    disabled={loading}
                    style={{ marginBottom: '1rem' }}
                  />

                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                    <Button renderIcon={Search} onClick={() => handleQuery('standard')} disabled={loading}>
                      Standard RAG
                    </Button>
                    <Button renderIcon={Chat} onClick={() => handleQuery('workflow')} disabled={loading} kind="secondary">
                      Workflow Agent
                    </Button>
                    <Button renderIcon={Compare} onClick={() => handleQuery('both')} disabled={loading} kind="tertiary">
                      Compare Both
                    </Button>
                    {sessionId && (
                      <Button renderIcon={Renew} onClick={handleResetSession} disabled={loading} kind="danger--tertiary">
                        Reset Session
                      </Button>
                    )}
                  </div>

                  {loading && <Loading description="Processing query..." withOverlay={false} />}
                  {error && (
                    <InlineNotification
                      kind="error"
                      title="Error"
                      subtitle={error}
                      onCloseButtonClick={() => setError(null)}
                    />
                  )}
                </Tile>

                {/* Results */}
                {(standardResult || workflowResult) && (
                  <Tabs selectedIndex={selectedTab} onChange={({ selectedIndex }) => setSelectedTab(selectedIndex)}>
                    <TabList aria-label="Results tabs">
                      <Tab disabled={!standardResult}>Standard RAG</Tab>
                      <Tab disabled={!workflowResult}>Workflow Agent</Tab>
                      <Tab disabled={!standardResult || !workflowResult}>Side-by-Side</Tab>
                      <Tab disabled={!metrics}>Metrics</Tab>
                    </TabList>
                    <TabPanels>
                      <TabPanel>
                        {standardResult && (
                          <Tile>
                            <h4>Standard RAG Response</h4>
                            <div style={{ background: '#262626', padding: '1rem', borderRadius: '4px', marginBottom: '1rem' }}>
                              <div
                                style={{ whiteSpace: 'pre-wrap' }}
                                dangerouslySetInnerHTML={{ __html: renderMarkdown(standardResult.answer) }}
                              />
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <Tag type="blue">Sources: {standardResult.num_sources}</Tag>
                              <Tag type="green">Time: {standardResult.total_time.toFixed(2)}s</Tag>
                            </div>
                          </Tile>
                        )}
                      </TabPanel>

                      <TabPanel>
                        {workflowResult && (
                          <Tile>
                            <h4>Workflow Agent Response</h4>
                            <div style={{ background: '#262626', padding: '1rem', borderRadius: '4px', marginBottom: '1rem' }}>
                              <div
                                style={{ whiteSpace: 'pre-wrap' }}
                                dangerouslySetInnerHTML={{ __html: renderMarkdown(workflowResult.answer) }}
                              />
                            </div>
                            
                            {workflowResult.workflow_status && (
                              <div style={{ marginTop: '1.5rem' }}>
                                <h5>Workflow Progress</h5>
                                <ProgressIndicator currentIndex={workflowResult.workflow_status.current_step - 1} vertical>
                                  {Array.from({ length: workflowResult.workflow_status.total_steps }, (_, i) => (
                                    <ProgressStep
                                      key={i}
                                      label={`Step ${i + 1}`}
                                      complete={workflowResult.workflow_status.completed_steps.includes(i)}
                                    />
                                  ))}
                                </ProgressIndicator>
                              </div>
                            )}
                            
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '1rem' }}>
                              <Tag type="purple">Status: {workflowResult.status}</Tag>
                              <Tag type="green">Time: {workflowResult.total_time.toFixed(2)}s</Tag>
                            </div>
                          </Tile>
                        )}
                      </TabPanel>

                      <TabPanel>
                        <Grid>
                          <Column lg={8}>
                            {standardResult && (
                              <Tile>
                                <h5>Standard RAG</h5>
                                <div style={{ background: '#262626', padding: '1rem', borderRadius: '4px', marginBottom: '1rem', maxHeight: '400px', overflowY: 'auto' }}>
                                  <div
                                    style={{ fontSize: '0.875rem', whiteSpace: 'pre-wrap', margin: 0 }}
                                    dangerouslySetInnerHTML={{ __html: renderMarkdown(standardResult.answer) }}
                                  />
                                </div>
                                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                  <Tag type="blue">Sources: {standardResult.num_sources}</Tag>
                                  <Tag type="green">Time: {standardResult.total_time.toFixed(2)}s</Tag>
                                </div>
                              </Tile>
                            )}
                          </Column>
                          <Column lg={8}>
                            {workflowResult && (
                              <Tile>
                                <h5>Workflow Agent</h5>
                                <div style={{ background: '#262626', padding: '1rem', borderRadius: '4px', marginBottom: '1rem', maxHeight: '400px', overflowY: 'auto' }}>
                                  <div
                                    style={{ fontSize: '0.875rem', whiteSpace: 'pre-wrap', margin: 0 }}
                                    dangerouslySetInnerHTML={{ __html: renderMarkdown(workflowResult.answer) }}
                                  />
                                </div>
                                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                  <Tag type="purple">Status: {workflowResult.status}</Tag>
                                  <Tag type="green">Time: {workflowResult.total_time.toFixed(2)}s</Tag>
                                </div>
                              </Tile>
                            )}
                          </Column>
                        </Grid>
                      </TabPanel>

                      <TabPanel>
                        {metrics && (
                          <Tile>
                            <h4>Performance Metrics</h4>
                            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                              <Tag type="blue">Total Queries: {metrics.total_queries}</Tag>
                              <Tag type="green">Standard: {metrics.standard_rag_queries}</Tag>
                              <Tag type="purple">Workflow: {metrics.workflow_agent_queries}</Tag>
                            </div>
                          </Tile>
                        )}
                      </TabPanel>
                    </TabPanels>
                  </Tabs>
                )}
              </>
            )}
          </Column>
        </Grid>
      </Content>
    </Theme>
  );
}

export default App;

// Made with Bob
