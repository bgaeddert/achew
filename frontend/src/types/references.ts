import type { BasicChapter } from './chapter';

export type ChapterRefType = 'abs' | 'embedded' | 'audnexus' | 'file_data' | 'json' | 'csv' | 'cue' | 'snapshot';
export type ReferenceValidationType = ChapterRefType | 'intelligent_detection';
export type TitleRefType = 'text' | 'epub' | 'custom';

interface ReferenceBase {
  id: string;
  name: string;
  short_name: string;
  description: string;
  metadata: Record<string, string>;
}

export interface ChapterReference extends ReferenceBase {
  type: ChapterRefType;
  chapters: BasicChapter[];
  duration: number;
}

export interface ReferenceValidationChapter extends BasicChapter {
  headings: string;
  valid: boolean;
}

export interface ReferenceValidationResult extends ReferenceBase {
  type: ReferenceValidationType;
  chapters: ReferenceValidationChapter[];
  duration: number;
}

export interface TitleReference extends ReferenceBase {
  type: TitleRefType;
  titles: string[];
}

export type Reference = ChapterReference | TitleReference;
