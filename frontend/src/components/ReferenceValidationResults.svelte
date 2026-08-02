<script lang="ts">
  import { onMount } from 'svelte';
  import ExternalLink from '@lucide/svelte/icons/external-link';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tooltip } from '../actions/tooltip';
  import { session } from '../stores/session';
  import type { ReferenceValidationResult } from '../types/references';
  import { api } from '../utils/api';
  import ChapterModal from './ChapterModal.svelte';

  let loading = $state(true);
  let returning = $state(false);
  let transcribing = $state(false);
  let quickEditing = $state(false);
  let references = $state<ReferenceValidationResult[]>([]);
  let selectedReferenceId = $state('');
  let modalOpen = $state(false);
  let modalReference = $state<ReferenceValidationResult | null>(null);
  let selectedReference = $derived(references.find((reference) => reference.id === selectedReferenceId));
  let modalIsIntelligentDetection = $derived(modalReference?.type === 'intelligent_detection');
  let hasValidatedReferences = $derived(references.some((reference) => reference.type !== 'intelligent_detection'));

  function validCount(reference: ReferenceValidationResult): number {
    return reference.chapters.filter((chapter) => chapter.valid).length;
  }

  function chapterCountLabel(count: number): string {
    return `${count} ${count === 1 ? 'chapter' : 'chapters'}`;
  }

  function validPercentage(reference: ReferenceValidationResult): number {
    if (reference.chapters.length === 0) return 0;
    return Math.round((validCount(reference) / reference.chapters.length) * 100);
  }

  function openReference(reference: ReferenceValidationResult) {
    modalReference = reference;
    modalOpen = true;
  }

  function closeReference() {
    modalOpen = false;
    modalReference = null;
  }

  async function returnToDetection() {
    returning = true;
    try {
      await session.restartSession('intelligent_chapter_detection');
    } catch (error) {
      console.error('Failed to return to intelligent chapter detection:', error);
      returning = false;
    }
  }

  async function transcribeReference() {
    if (!selectedReference || transcribing || quickEditing) return;
    transcribing = true;
    try {
      await api.session.transcribeValidatedReference(selectedReference.id);
    } catch (error) {
      console.error('Failed to prepare reference for transcription:', error);
      session.setError('Failed to prepare reference for transcription: ' + (error as Error).message);
      transcribing = false;
    }
  }

  async function quickEditReference() {
    if (!selectedReference || quickEditing || transcribing) return;
    quickEditing = true;
    try {
      await api.session.startWorkflow('quick_edit', selectedReference.id);
    } catch (error) {
      console.error('Failed to quick edit validated reference:', error);
      session.setError('Failed to open validated reference in Quick Edit: ' + (error as Error).message);
      quickEditing = false;
    }
  }

  onMount(async () => {
    try {
      const response = await api.session.getReferenceValidationResults();
      references = response.references;
      const intelligentDetection = references.find((reference) => reference.type === 'intelligent_detection');
      if (intelligentDetection) {
        selectedReferenceId = intelligentDetection.id;
      }
    } catch (error) {
      console.error('Failed to load chapter reference validation results:', error);
      session.setError('Failed to load reference validation results: ' + (error as Error).message);
    } finally {
      loading = false;
    }
  });
</script>

<div class="validation-results">
  <div class="header">
    <h2>{hasValidatedReferences ? 'Reference Validation Results' : 'Intelligent Detection Results'}</h2>
    {#if hasValidatedReferences}
      <p>
        Compare Intelligent Chapter Detection with each timed Chapter Reference. The details show the spoken heading
        what was heard at each candidate timestamp and which headings the selected LLM considered valid chapter
        markers.
        Select an option to use its resulting chapter list on the Transcribe Titles page, or open the selected result
        directly in Quick Edit.
      </p>
    {:else}
      <p>
        Review the chapter headings accepted by Intelligent Chapter Detection, then continue with those boundaries on
        the Transcribe Titles page or open them directly in Quick Edit.
      </p>
    {/if}
  </div>

  {#if loading}
    <div class="loading-state">
      <span class="spinner"></span>
      Loading validation results…
    </div>
  {:else if references.length === 0}
    <div class="empty-state">
      <TriangleAlert size="20" />
      <span>No timed chapter references were validated.</span>
    </div>
  {:else}
    <div class="results-list">
      {#each references as reference (reference.id)}
        <div class="option-card" class:selected={selectedReferenceId === reference.id}>
          <label>
            <div class="option-layout">
              <input
                type="radio"
                name="validated-reference"
                value={reference.id}
                bind:group={selectedReferenceId}
                disabled={transcribing || quickEditing || returning}
              />
              <div class="option-content">
                <div class="option-header">
                  <b>{reference.name}</b>
                  <div class="chapter-count-container">
                    {#if reference.type === 'intelligent_detection'}
                      <button
                        class="chapter-count clickable"
                        onclick={() => openReference(reference)}
                        use:tooltip={'Click to view accepted chapter headings'}
                      >
                        {validCount(reference)} valid
                        <ExternalLink size="12" />
                      </button>
                    {:else}
                      <button
                        class="chapter-count clickable"
                        onclick={() => openReference(reference)}
                        use:tooltip={'Click to view validation details'}
                      >
                        {chapterCountLabel(reference.chapters.length)}
                        <ExternalLink size="12" />
                      </button>
                      <span class="chapter-count">{validCount(reference)} of {reference.chapters.length} valid</span>
                      <span class="chapter-count percentage-valid">{validPercentage(reference)}% valid</span>
                    {/if}
                  </div>
                </div>
                <p class="description">{reference.description}</p>
              </div>
            </div>
          </label>
        </div>
      {/each}
    </div>
  {/if}

  <div class="actions">
    <button class="btn btn-outline" onclick={returnToDetection} disabled={returning || transcribing || quickEditing}>
      {returning ? 'Returning…' : 'Back to Intelligent Chapter Detection'}
    </button>
    {#if references.length > 0}
      <button
        class="btn btn-outline"
        onclick={quickEditReference}
        disabled={!selectedReference || returning || transcribing || quickEditing}
      >
        {quickEditing ? 'Opening…' : 'Quick Edit'}
      </button>
    {/if}
    <button
      class="btn btn-verify"
      onclick={transcribeReference}
      disabled={!selectedReference || returning || transcribing || quickEditing}
    >
      {transcribing ? 'Opening…' : 'Transcribe'}
    </button>
  </div>
</div>

<ChapterModal
  bind:isOpen={modalOpen}
  title={modalReference?.name ?? ''}
  duration={modalReference?.duration}
  chapters={modalReference?.chapters ?? []}
  headingsOnly={modalIsIntelligentDetection}
  onclose={closeReference}
/>

<style>
  .validation-results {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }

  .header {
    max-width: 720px;
    margin: 0 auto 2rem;
    text-align: center;
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

  .results-list {
    display: grid;
    gap: 1rem;
  }

  .option-card {
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 0;
    background: var(--bg-secondary);
    transition: all 0.1s ease;
    cursor: pointer;
  }

  .option-card:hover {
    border-color: var(--primary-hover);
  }

  .option-card.selected {
    border-color: var(--primary-color);
  }

  .option-card label {
    display: block;
    padding: 0.75rem;
    margin: 0;
    cursor: pointer;
  }

  .option-card input[type='radio'] {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    margin: 0 0.5rem;
    accent-color: var(--primary-contrast);
    cursor: pointer;
  }

  .option-layout {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 100%;
  }

  .option-content {
    min-width: 0;
  }

  .option-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
    flex-wrap: wrap;
  }

  .chapter-count-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .chapter-count {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.75rem;
    border: 1px solid transparent;
    border-radius: 60px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    font-size: 0.75rem;
    line-height: normal;
    cursor: default;
  }

  .chapter-count.clickable {
    border-color: color-mix(in srgb, var(--primary-color) 35%, transparent);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .chapter-count.clickable:hover {
    color: white;
    background: var(--primary-color);
    transform: translateY(-1px);
  }

  .chapter-count.clickable:active {
    transform: translateY(0);
  }

  .chapter-count.percentage-valid {
    color: var(--success);
  }

  .description {
    margin: 0;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  .loading-state,
  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    min-height: 8rem;
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    background: var(--bg-secondary);
    color: var(--text-secondary);
  }

  .spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--border-color);
    border-top-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .actions {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 2rem;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 600px) {
    .option-card {
      padding: 0;
    }

    .option-card label {
      padding: 1rem;
    }

    .actions {
      flex-direction: column-reverse;
      align-items: stretch;
    }
  }
</style>
