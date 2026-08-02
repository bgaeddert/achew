<script lang="ts">
  import IntelligentChapterDetection from './IntelligentChapterDetection.svelte';

  interface Props {
    onsettingscomplete?: () => void;
  }

  let { onsettingscomplete }: Props = $props();
</script>

<IntelligentChapterDetection settingsMode={true} {onsettingscomplete} />
