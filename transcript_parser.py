import argparse
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TranscriptParser:
    def __init__(self, min_segment_length: int = 20):
        """
        Initialize the TranscriptParser
        
        Args:
            min_segment_length: Minimum character length to consider a segment as a paragraph
        """
        self.min_segment_length = min_segment_length

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        return text.strip()

    def parse_json_file(self, json_path: Path) -> dict:
        """Parse a Whisper JSON transcript file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Calculate duration from segments
            segments = data.get('segments', [])
            if segments:
                duration = int(segments[-1].get('end', 0))
                minutes = duration // 60
                seconds = duration % 60
                duration_str = f"{minutes}:{seconds:02d}"
            else:
                duration_str = "Unknown"

            # Get metadata from txt file for upload date
            txt_path = json_path.with_suffix('.txt')
            upload_date = None
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('Upload Date:'):
                            upload_date = line.split(':', 1)[1].strip()
                            break

            # Get metadata from filename
            metadata = {
                'Title': json_path.stem,
                'URL': f"https://www.youtube.com/watch?v={json_path.stem.split('_')[-1]}",
                'Language': data.get('language', 'en'),
                'Duration': duration_str,
                'Upload Date': upload_date or 'Unknown'
            }

            # Get text from segments in order and join them
            segments = sorted(segments, key=lambda x: x.get('start', 0))
            full_text = ' '.join(segment.get('text', '').strip() for segment in segments)

            # Split into paragraphs (roughly every 500 characters at sentence boundaries)
            paragraphs = []
            current = []
            current_length = 0
            
            for sentence in full_text.split('. '):
                if not sentence.strip():
                    continue
                
                current.append(sentence.strip())
                current_length += len(sentence)
                
                # Start new paragraph after ~500 characters at sentence boundary
                if current_length > 500:
                    paragraphs.append('. '.join(current) + '.')
                    current = []
                    current_length = 0
            
            # Add any remaining content
            if current:
                paragraphs.append('. '.join(current) + '.')

            return {
                'metadata': metadata,
                'paragraphs': paragraphs
            }

        except Exception as e:
            logging.error(f"Error parsing JSON {json_path}: {e}")
            return None

    def save_parsed_transcript(self, parsed_content: dict, output_path: Path):
        """Save parsed content to a clean format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata
                for key, value in parsed_content['metadata'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Write main transcript text
                if parsed_content['paragraphs']:
                    f.write("Transcription:\n\n")
                    for paragraph in parsed_content['paragraphs']:
                        f.write(paragraph + "\n\n")

        except Exception as e:
            logging.error(f"Error saving to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Parse and clean transcript files')
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('--output-dir', help='Output directory (default: input directory)')
    parser.add_argument('--min-length', type=int, default=20,
                      help='Minimum character length for paragraphs (default: 20)')
    parser.add_argument('--suffix', default='_clean',
                      help='Suffix to add to output files (default: _clean)')

    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input path: {input_path}")
    logging.info(f"Output directory: {output_dir}")

    transcript_parser = TranscriptParser(min_segment_length=args.min_length)

    def process_file(file_path: Path):
        logging.info(f"Checking file: {file_path}")
        if not file_path.suffix == '.json':
            logging.info(f"Skipping non-JSON file: {file_path}")
            return

        output_path = output_dir / f"{file_path.stem.replace('_clean', '')}{args.suffix}.txt"
        
        logging.info(f"Processing: {file_path}")
        parsed_content = transcript_parser.parse_json_file(file_path)
        
        if parsed_content:
            transcript_parser.save_parsed_transcript(parsed_content, output_path)
            logging.info(f"Saved cleaned transcript to: {output_path}")
        else:
            logging.error(f"Failed to parse content from {file_path}")

    if input_path.is_dir():
        logging.info(f"Searching for JSON files in: {input_path}")
        json_files = list(input_path.glob('*.json'))
        logging.info(f"Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            if '_clean' not in file_path.stem:
                process_file(file_path)
            else:
                logging.info(f"Skipping already cleaned file: {file_path}")
    else:
        process_file(input_path)

if __name__ == "__main__":
    main() 