<script lang="ts">
  import { onMount } from 'svelte';
  import { tooltip } from '../actions/tooltip';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import CircleQuestionMark from '@lucide/svelte/icons/circle-question-mark';
  import Eye from '@lucide/svelte/icons/eye';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import ChapterModal from './ChapterModal.svelte';
  import { session } from '../stores/session';
  import type { ChapterReference } from '../types/references';
  import type { LLMModel, LLMProvider } from '../types/api';
  import { api } from '../utils/api';

  let loading = $state(false);
  let loadingModels = $state(false);
  let loadingOptions = $state(false);
  let updatingTerms = $state(false);
  let providers = $state<LLMProvider[]>([]);
  let models = $state<LLMModel[]>([]);
  let providerId = $state('');
  let modelId = $state('');
  let chapterRefs = $state<ChapterReference[]>([]);
  let referenceId = $state('');
  let baseTerms = $state<string[]>([]);
  let voskTerms = $state('');
  let minimumPauseSeconds = $state(2);
  let advancedExpanded = $state(false);
  let showReferenceChapters = $state(false);
  let referenceChaptersTitle = $state('');
  let referenceChapters = $state<Array<{ timestamp: number; title: string }>>([]);

  let configuredProviders = $derived(providers.filter((provider) => provider.is_enabled && provider.is_configured));
  let currentReference = $derived(chapterRefs.find((reference) => reference.id === referenceId));
  let canRun = $derived(
    Boolean(providerId && modelId && voskTerms.trim() && !loading && !loadingModels && !updatingTerms),
  );
  let canUpdateTerms = $derived(
    Boolean(providerId && modelId && referenceId && !loading && !loadingModels && !updatingTerms),
  );

  function parsedTerms() {
    return voskTerms
      .toLowerCase()
      .split(/[\s,]+/)
      .map((term) => term.replace(/[^a-z0-9']/g, ''))
      .filter(Boolean);
  }

  function resetTerms() {
    voskTerms = baseTerms.join(' ');
  }

  function viewReferenceChapters() {
    if (!currentReference) return;
    referenceChaptersTitle = currentReference.name;
    referenceChapters = currentReference.chapters.map((chapter) => ({
      timestamp: chapter.timestamp,
      title: chapter.title || 'No Title',
    }));
    showReferenceChapters = true;
  }

  function closeReferenceChapters() {
    showReferenceChapters = false;
    referenceChaptersTitle = '';
    referenceChapters = [];
  }

  async function loadModels(nextProviderId: string) {
    if (!nextProviderId) {
      models = [];
      modelId = '';
      return;
    }
    loadingModels = true;
    try {
      const response = await api.llm.getModels(nextProviderId);
      models = response.models;
      modelId = models.find((model) => model.id === modelId)?.id ?? models[0]?.id ?? '';
    } catch (error) {
      console.error('Failed to load LLM models:', error);
      models = [];
      modelId = '';
    } finally {
      loadingModels = false;
    }
  }

  async function loadConfiguration() {
    loadingOptions = true;
    try {
      const [providerResponse, savedOptions, options] = await Promise.all([
        api.llm.getProviders(),
        api.batch.getAIOptions(),
        api.session.getIntelligentChapterDetectionOptions(),
      ]);
      providers = providerResponse.providers;
      chapterRefs = options.chapter_refs;
      baseTerms = options.base_terms;
      resetTerms();
      referenceId = chapterRefs[0]?.id ?? '';

      const availableProviders = providerResponse.providers.filter(
        (provider) => provider.is_enabled && provider.is_configured,
      );
      providerId =
        availableProviders.find((provider) => provider.id === savedOptions.provider_id)?.id ??
        availableProviders[0]?.id ??
        '';
      modelId = savedOptions.model_id;
      await loadModels(providerId);
    } catch (error) {
      console.error('Failed to load intelligent detection configuration:', error);
      session.setError('Failed to load intelligent detection options: ' + (error as Error).message);
    } finally {
      loadingOptions = false;
    }
  }

  async function saveSelection() {
    try {
      const options = await api.batch.getAIOptions();
      await api.batch.updateAIOptions({ ...options, provider_id: providerId, model_id: modelId });
    } catch (error) {
      console.warn('Failed to save LLM selection:', error);
    }
  }

  async function handleProviderChange(event: Event) {
    providerId = (event.target as HTMLSelectElement).value;
    modelId = '';
    await loadModels(providerId);
    await saveSelection();
  }

  async function handleModelChange(event: Event) {
    modelId = (event.target as HTMLSelectElement).value;
    await saveSelection();
  }

  async function updateSearchTerms() {
    if (!canUpdateTerms) return;
    updatingTerms = true;
    try {
      await saveSelection();
      const response = await api.session.updateIntelligentSearchTerms(providerId, modelId, referenceId, parsedTerms());
      voskTerms = response.terms.join(' ');
    } catch (error) {
      console.error('Failed to update Vosk search terms:', error);
      session.setError('Failed to update search terms: ' + (error as Error).message);
    } finally {
      updatingTerms = false;
    }
  }

  async function run() {
    if (!canRun) return;
    loading = true;
    try {
      await saveSelection();
      await api.session.intelligentChapterDetection('run', providerId, modelId, {
        voskTerms: parsedTerms(),
        minimumPauseSeconds,
      });
    } catch (error) {
      console.error('Failed to run intelligent chapter detection:', error);
      session.setError('Failed to start intelligent detection: ' + (error as Error).message);
      loading = false;
    }
  }

  async function cancel() {
    if (loading || updatingTerms) return;
    loading = true;
    try {
      await api.session.intelligentChapterDetection('skip');
    } catch (error) {
      console.error('Failed to cancel intelligent chapter detection:', error);
      session.setError('Failed to cancel intelligent detection: ' + (error as Error).message);
      loading = false;
    }
  }

  onMount(loadConfiguration);
</script>

<div class="intelligent-detection">
  <div class="header">
    <h2>Intelligent Chapter Detection</h2>
    <p>
      Check the strongest pauses for spoken headings, then use the selected LLM to choose the most likely chapter
      boundaries. Especially useful for stubborn audiobooks with too many possible pauses.
    </p>
  </div>

  <section class="configuration">
    {#if configuredProviders.length === 0 && !loadingOptions}
      <div class="notice">
        <TriangleAlert size="20" />
        <div>
          <strong>No LLM provider is ready.</strong>
          <span>Configure and enable a provider in Settings to use intelligent detection.</span>
        </div>
      </div>
    {:else}
      <div class="form-row">
        <label>
          <span>Provider</span>
          <div class="select-wrap">
            <select value={providerId} onchange={handleProviderChange} disabled={loading || loadingOptions}>
              {#each configuredProviders as provider}
                <option value={provider.id}>{provider.name}</option>
              {/each}
            </select>
            <ChevronDown size={18} strokeWidth={2.25} />
          </div>
        </label>
        <label>
          <span>Model</span>
          <div class="select-wrap">
            <select
              value={modelId}
              onchange={handleModelChange}
              disabled={loading || loadingOptions || loadingModels || models.length === 0}
            >
              {#if loadingModels}
                <option>Loading models…</option>
              {:else if models.length === 0}
                <option>No models available</option>
              {:else}
                {#each models as model}
                  <option value={model.id}>{model.name}</option>
                {/each}
              {/if}
            </select>
            <ChevronDown size={18} strokeWidth={2.25} />
          </div>
        </label>
      </div>

      <div class="reference-row">
        <label for="heading-reference">Reference</label>
        <div class="reference-controls">
          <div class="select-wrap reference-select-wrap">
            <select
              id="heading-reference"
              bind:value={referenceId}
              disabled={loading || updatingTerms || chapterRefs.length === 0}
            >
              {#if chapterRefs.length === 0}
                <option value="">No chapter references available</option>
              {:else}
                {#each chapterRefs as reference (reference.id)}
                  <option value={reference.id}>{reference.name} ({reference.chapters.length} chapters)</option>
                {/each}
              {/if}
            </select>
            <ChevronDown size={18} strokeWidth={2.25} />
          </div>
          <button
            type="button"
            class="view-reference-btn"
            onclick={viewReferenceChapters}
            disabled={!currentReference}
            aria-label="View chapters from selected Reference"
            use:tooltip={'View chapters from selected Reference'}
          >
            <Eye size="20" />
          </button>
          <button class="btn btn-outline update-terms-btn" onclick={updateSearchTerms} disabled={!canUpdateTerms}>
            {updatingTerms ? 'Updating…' : 'Update Search Terms'}
          </button>
        </div>
        {#if currentReference}
          <p>Optional: Use this reference to suggest additional spoken heading words. No audio is sent to the LLM.</p>
        {/if}
      </div>

      <div class="terms-input-container">
        <div class="terms-heading">
          <label for="vosk-terms">Vosk Search Terms</label>
        </div>
        <div class="input-with-reset">
          <textarea
            id="vosk-terms"
            bind:value={voskTerms}
            disabled={loading || updatingTerms}
            class="terms-input"
            placeholder="Enter Vosk search terms…"
            rows="3"
          ></textarea>
          <button
            type="button"
            class="reset-button"
            onclick={resetTerms}
            aria-label="Reset to default Vosk terms"
            title="Reset to default Vosk terms"
            disabled={loading || updatingTerms}
          >
            <RotateCcw size="12" />
          </button>
        </div>
      </div>

      <div class="advanced-section">
        <button class="advanced-toggle" onclick={() => (advancedExpanded = !advancedExpanded)} type="button">
          Advanced
          <span class="chevron" class:expanded={advancedExpanded}><ChevronDown size="12" /></span>
        </button>
        {#if advancedExpanded}
          <div class="advanced-panel">
            <div class="setting-item">
              <div class="setting-header">
                <label for="minimum-pause">Minimum Pause</label>
                <span
                  class="help-icon"
                  use:tooltip={{ text: 'Only pauses at or above this duration are checked with Vosk.', delay: 0 }}
                >
                  <CircleQuestionMark size="14" />
                </span>
              </div>
              <div class="slider-container">
                <input
                  id="minimum-pause"
                  type="range"
                  min="2"
                  max="6"
                  step="0.5"
                  bind:value={minimumPauseSeconds}
                  class="slider"
                  disabled={loading || updatingTerms}
                />
                <div class="slider-value">{minimumPauseSeconds}s</div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </section>

  <div class="actions">
    <button class="btn btn-cancel" onclick={cancel} disabled={loading || updatingTerms}>Cancel</button>
    <button class="btn btn-verify" onclick={run} disabled={!canRun}>
      {loading ? 'Starting…' : 'Detect Chapters'}
    </button>
  </div>
</div>

<ChapterModal
  bind:isOpen={showReferenceChapters}
  title={referenceChaptersTitle}
  chapters={referenceChapters}
  onclose={closeReferenceChapters}
/>

<style>
  .intelligent-detection {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }
  .header {
    text-align: center;
    margin: 0 auto 3rem;
    max-width: 680px;
  }
  .header h2 {
    margin: 0 0 0.75rem;
    font-size: 2rem;
    font-weight: 600;
  }
  .header p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.5;
  }
  .configuration {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
  }
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  label {
    display: grid;
    gap: 0.5rem;
    color: var(--text-primary);
    font-size: 0.9rem;
    font-weight: 500;
  }
  .select-wrap {
    position: relative;
  }
  select {
    width: 100%;
    min-width: 0;
    appearance: none;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 0.5rem 2.25rem 0.5rem 0.75rem;
    font: inherit;
    font-size: 0.9rem;
    line-height: 1.2;
    cursor: pointer;
    transition: border-color 0.2s ease;
  }
  select:hover:not(:disabled),
  select:focus {
    border-color: var(--primary-color);
  }
  select:focus {
    outline: none;
  }
  select:disabled {
    cursor: wait;
    opacity: 0.6;
  }
  .select-wrap :global(svg) {
    position: absolute;
    right: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-secondary);
    pointer-events: none;
  }
  .notice {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    color: #b7791f;
    background: color-mix(in srgb, #f59e0b 12%, var(--bg-card));
    border-radius: 8px;
    padding: 1rem;
  }
  .notice div {
    display: grid;
    gap: 0.25rem;
  }
  .notice span {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.4;
  }
  .reference-row {
    display: grid;
    gap: 0.5rem;
    margin-top: 1.25rem;
  }
  .reference-controls {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 0.5rem;
    align-items: stretch;
  }
  .reference-select-wrap {
    flex: 1;
    min-width: 0;
  }
  .update-terms-btn {
    min-height: 2.4rem;
    padding: 0.5rem 0.75rem;
    white-space: nowrap;
  }
  .view-reference-btn {
    flex-shrink: 0;
    width: 2.4rem;
    min-height: 2.4rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    background: var(--bg-primary);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }
  .view-reference-btn:hover:not(:disabled) {
    background: var(--hover-bg);
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }
  .view-reference-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .reference-row p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.4;
  }
  .terms-input-container {
    margin-top: 1rem;
  }
  .terms-heading {
    margin-bottom: 0.5rem;
  }
  .input-with-reset {
    position: relative;
    display: flex;
    align-items: flex-start;
  }
  .terms-input {
    flex: 1;
    padding: 0.5rem 2.5rem 0.5rem 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.9rem;
    font-family: inherit;
    resize: vertical;
    min-height: 7.75rem;
    line-height: 1.4;
    transition: border-color 0.2s ease;
  }
  .terms-input:focus {
    outline: none;
    border-color: var(--primary-color);
  }
  .terms-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .reset-button {
    position: absolute;
    right: 0.5rem;
    top: 0.5rem;
    width: 1.5rem;
    height: 1.5rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }
  .reset-button:hover {
    background: var(--bg-tertiary);
    color: var(--primary-color);
  }
  .reset-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .advanced-section {
    margin-top: 1rem;
  }
  .advanced-toggle {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    border: 0;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0.35rem;
    font: inherit;
  }
  .chevron {
    display: inline-flex;
    transition: transform 0.16s ease;
  }
  .chevron.expanded {
    transform: rotate(180deg);
  }
  .advanced-panel {
    padding: 0.8rem 0 0;
  }
  .setting-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
  }
  .setting-header label {
    color: var(--text-primary);
  }
  .help-icon {
    display: inline-flex;
    color: var(--text-secondary);
    cursor: help;
  }
  .slider-container {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.75rem;
    max-width: 420px;
    margin: 0.85rem auto 0;
  }
  .slider {
    width: 100%;
    accent-color: var(--primary-color);
  }
  .slider-value {
    min-width: 2.5rem;
    color: var(--text-primary);
    font-weight: 600;
  }
  .actions {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 2rem;
  }
  @media (max-width: 600px) {
    .form-row {
      grid-template-columns: 1fr;
    }
    .reference-controls {
      grid-template-columns: minmax(0, 1fr) auto;
    }
    .update-terms-btn {
      grid-column: 1 / -1;
      width: 100%;
    }
    .actions {
      flex-direction: column-reverse;
      align-items: stretch;
    }
    .actions button {
      width: 100%;
    }
  }
</style>
