from pytubefix import YouTube, Channel
import whisper
import os
import ffmpeg
import json
import torch
from datetime import datetime
from pathlib import Path
import time
import concurrent.futures
from typing import Optional, List, Dict
import logging
from tqdm import tqdm  # Add this import for progress bars
import argparse  # Add this import at the top
from tenacity import retry, stop_after_attempt, wait_exponential
import sys
import signal
import googleapiclient.discovery
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('channel_transcriber.log'),
        logging.StreamHandler()
    ]
)

class ChannelTranscriber:
    def __init__(
        self,
        output_base_dir: str = "channel_data",
        max_workers: int = 2,
        device: str = "auto",
        file_mode: str = "skip",
        whisper_model: str = "turbo",
        keep_video: bool = False
    ):
        self.output_base_dir = Path(output_base_dir)
        self.max_workers = max_workers
        self.device = self._setup_device(device)
        self.file_mode = file_mode
        self.whisper_model = whisper_model
        self.keep_video = keep_video
        
        # Initialize directories
        self.videos_dir = self.output_base_dir / "videos"
        self.audio_dir = self.output_base_dir / "audio"
        self.transcripts_dir = self.output_base_dir / "transcripts"
        self.metadata_dir = self.output_base_dir / "metadata"
        
        for directory in [self.videos_dir, self.audio_dir, 
                         self.transcripts_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Load Whisper model
        self.model = self._load_whisper_model()
        
        self._interrupt_received = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Initialize YouTube API
        load_dotenv()
        self.youtube_api = self._setup_youtube_api()
        
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return device
    
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model with error handling"""
        try:
            logging.info(f"Loading Whisper model '{self.whisper_model}' on {self.device}")
            return whisper.load_model(self.whisper_model).to(self.device)
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    
    def clean_filename(self, filename: str) -> str:
        return ''.join(c if c.isalnum() or c in '.-_ ' else '_' for c in filename)
    
    def get_video_info(self, url: str) -> Dict:
        """Get basic video information without downloading"""
        try:
            yt = YouTube(url)
            return {
                'title': yt.title,
                'url': url,
                'video_id': yt.video_id,
                'length': yt.length,
                'publish_date': str(yt.publish_date),
                'views': yt.views,
                'author': yt.author,
                'description': yt.description
            }
        except Exception as e:
            logging.error(f"Error getting video info for {url}: {str(e)}")
            return None

    def get_existing_transcripts(self) -> Dict[str, Dict]:
        """Get a mapping of video IDs to existing transcript information"""
        existing = {}
        for json_file in self.transcripts_dir.glob("*_*.json"):
            try:
                # Extract video_id from filename (assumes format: title_videoId.json)
                video_id = json_file.stem.split('_')[-1]
                
                # Get corresponding txt file
                txt_file = self.transcripts_dir / f"{json_file.stem}.txt"
                
                if txt_file.exists():
                    existing[video_id] = {
                        'json_path': json_file,
                        'txt_path': txt_file,
                        'processed_at': datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
                    }
            except Exception as e:
                logging.warning(f"Error reading existing transcript {json_file}: {e}")
        return existing

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _download_video(self, yt: YouTube, output_path: str, filename: str):
        """Download video with retry mechanism"""
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if not stream:
            raise ValueError("No suitable video stream found")
        stream.download(output_path=output_path, filename=filename)

    def process_video(self, video_url: str) -> Dict:
        """Process a single video: download, convert to audio, and transcribe"""
        try:
            # Get video info
            video_info = self.get_video_info(video_url)
            if not video_info:
                return None
            
            video_id = video_info['video_id']
            clean_title = self.clean_filename(video_info['title'])
            
            # Set up file paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = self.videos_dir / f"{clean_title}_{video_id}.mp4"
            audio_path = self.audio_dir / f"{clean_title}_{video_id}.mp3"
            transcript_json = self.transcripts_dir / f"{clean_title}_{video_id}.json"
            transcript_txt = self.transcripts_dir / f"{clean_title}_{video_id}.txt"
            
            # Skip if already processed
            if self.file_mode == 'skip' and transcript_json.exists():
                logging.info(f"Already processed video: {clean_title}")
                return None
            
            # Download video
            logging.info(f"Downloading: {clean_title}")
            yt = YouTube(video_url)
            self._download_video(yt, str(self.videos_dir), f"{clean_title}_{video_id}.mp4")
            
            # Convert to audio
            logging.info(f"Converting to audio: {clean_title}")
            input_stream = ffmpeg.input(str(video_path))
            output_stream = ffmpeg.output(input_stream, str(audio_path), acodec='libmp3lame', q=0)
            ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True)
            
            # Transcribe
            logging.info(f"Transcribing: {clean_title}")
            result = self.model.transcribe(str(audio_path))
            
            # Save transcripts
            with open(transcript_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            with open(transcript_txt, 'w', encoding='utf-8') as f:
                f.write(f"Title: {video_info['title']}\n")
                f.write(f"URL: {video_url}\n")
                f.write(f"Author: {video_info['author']}\n")
                f.write(f"Published: {video_info['publish_date']}\n")
                f.write(f"Language: {result['language']}\n")
                f.write("\nTranscription:\n")
                f.write(result['text'])
                
                f.write("\n\nDetailed segments with timestamps:\n")
                for segment in result['segments']:
                    start_time = int(segment['start'])
                    end_time = int(segment['end'])
                    text = segment['text']
                    f.write(f"\n[{start_time//60}:{start_time%60:02d} - {end_time//60}:{end_time%60:02d}] {text}")
            
            # Cleanup
            if not self.keep_video and video_path.exists():
                video_path.unlink()
            
            return {
                'video_info': video_info,
                'files': {
                    'video': str(video_path),
                    'audio': str(audio_path),
                    'transcript_json': str(transcript_json),
                    'transcript_txt': str(transcript_txt)
                },
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error processing video {video_url}: {str(e)}")
            return {
                'video_url': video_url,
                'error': str(e),
                'success': False
            }

    def _setup_youtube_api(self):
        """Initialize YouTube API client"""
        try:
            api_key = os.environ.get("YOUTUBE_API_KEY")
            if not api_key:
                logging.warning("YouTube API key not found in environment variables")
                return None
                
            return googleapiclient.discovery.build(
                "youtube", 
                "v3", 
                developerKey=api_key
            )
        except Exception as e:
            logging.warning(f"Failed to initialize YouTube API: {e}")
            return None

    def get_channel_videos(self, channel_url: str, max_results: int = None) -> List[Dict]:
        """Get video information using YouTube API instead of scraping"""
        if not self.youtube_api:
            logging.warning("YouTube API not available, falling back to scraping")
            return None
            
        try:
            # Extract channel ID from URL
            channel = Channel(channel_url)
            channel_id = channel.channel_id
            
            videos_data = []
            next_page_token = None
            total_retrieved = 0
            
            while True:
                # Create request with pagination
                request = self.youtube_api.search().list(
                    part="snippet",
                    channelId=channel_id,
                    maxResults=50,  # Maximum allowed per request
                    order="date",
                    type="video",
                    pageToken=next_page_token
                )

                response = request.execute()
                
                if not response.get('items'):
                    break
                    
                # Get video IDs for this page
                video_ids = [item['id']['videoId'] for item in response['items']]

                # Get video details including duration
                video_details = self.youtube_api.videos().list(
                    part="contentDetails",
                    id=','.join(video_ids)
                ).execute()

                # Create a duration lookup dictionary
                duration_lookup = {
                    item['id']: item['contentDetails']['duration']
                    for item in video_details['items']
                }

                # Process videos from this page
                for item in response['items']:
                    video_id = item['id']['videoId']
                    video_data = {
                        'title': item['snippet']['title'],
                        'duration': duration_lookup.get(video_id, 'Duration not available'),
                        'video_id': video_id,
                        'published_at': item['snippet']['publishedAt'],
                        'description': item['snippet']['description'],
                        'channel_title': item['snippet']['channelTitle'],
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    }
                    videos_data.append(video_data)
                    total_retrieved += 1

                    if max_results and total_retrieved >= max_results:
                        break

                next_page_token = response.get('nextPageToken')
                if not next_page_token or (max_results and total_retrieved >= max_results):
                    break

            return videos_data
            
        except Exception as e:
            logging.error(f"Error fetching channel videos: {e}")
            return None

    def process_channel(self, channel_url: str, skip_existing: bool = True) -> List[Dict]:
        """Process all videos from a YouTube channel"""
        try:
            # Try to get videos using API first
            videos_data = self.get_channel_videos(channel_url)
            
            if videos_data:
                logging.info(f"Successfully retrieved {len(videos_data)} videos using YouTube API")
                videos_to_process = []
                skipped_videos = []
                
                # Get existing transcripts
                existing_transcripts = self.get_existing_transcripts() if skip_existing else {}
                
                total_duration = 0
                processed_duration = 0
                
                for video in tqdm(videos_data, desc="Checking videos"):
                    try:
                        video_id = video['video_id']
                        
                        # Add video to appropriate list
                        if skip_existing and video_id in existing_transcripts:
                            processed_duration += video['length']
                            skipped_videos.append({
                                'video_info': video,
                                'transcript_info': existing_transcripts[video_id],
                                'status': 'skipped'
                            })
                        else:
                            videos_to_process.append((video['url'], video))
                            
                        total_duration += video['length']
                            
                    except Exception as e:
                        logging.error(f"Error processing video info: {e}")
                
                total_videos = len(videos_to_process) + len(skipped_videos)
                total_hours = total_duration / 3600
                processed_hours = processed_duration / 3600
                remaining_hours = (total_duration - processed_duration) / 3600
                
                logging.info(f"\nChannel Statistics:")
                logging.info(f"Total videos: {total_videos}")
                logging.info(f"Total duration: {total_hours:.1f} hours")
                logging.info(f"Already processed: {processed_hours:.1f} hours")
                logging.info(f"Remaining to process: {remaining_hours:.1f} hours")
                logging.info(f"Videos to process: {len(videos_to_process)}")
                logging.info(f"Videos already processed: {len(skipped_videos)}")
                
                # Save channel metadata
                channel_name = self.clean_filename(videos_data[0]['channel_title'])
                channel_meta = {
                    'channel_name': videos_data[0]['channel_title'],
                    'channel_url': channel_url,
                    'video_count': total_videos,
                    'total_duration_seconds': total_duration,
                    'processed_duration_seconds': processed_duration,
                    'to_process': len(videos_to_process),
                    'skipped': len(skipped_videos),
                    'processed_at': datetime.now().isoformat(),
                }
                
                with open(self.metadata_dir / f"{channel_name}_metadata.json", 'w') as f:
                    json.dump(channel_meta, f, indent=2)
                
                # Process videos in parallel with progress bar
                results = skipped_videos.copy()  # Include skipped videos in results
                
                if videos_to_process:
                    # Calculate total duration of videos to process
                    remaining_duration = sum(info['length'] for _, info in videos_to_process)
                    processed_this_run = 0
                    
                    with tqdm(total=remaining_duration, desc="Processing videos", 
                             unit='s', unit_scale=True) as pbar:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            future_to_url = {
                                executor.submit(self.process_video, url): (url, info) 
                                for url, info in videos_to_process
                            }
                            
                            for future in concurrent.futures.as_completed(future_to_url):
                                url, info = future_to_url[future]
                                try:
                                    result = future.result()
                                    if result:
                                        results.append(result)
                                    processed_this_run += info['length']
                                    pbar.update(info['length'])
                                    hours_done = processed_this_run / 3600
                                    hours_left = (remaining_duration - processed_this_run) / 3600
                                    pbar.set_postfix({
                                        'video': info['title'][:30],
                                        'processed': f"{hours_done:.1f}h",
                                        'remaining': f"{hours_left:.1f}h"
                                    })
                                except Exception as e:
                                    logging.error(f"Error processing {url}: {str(e)}")
                                    results.append({
                                        'video_url': url,
                                        'video_info': info,
                                        'error': str(e),
                                        'success': False
                                    })
                                    pbar.update(info['length'])
                
                # Save overall results
                results_file = self.metadata_dir / f"{channel_name}_results.json"
                if results_file.exists():
                    # If results file exists, merge with previous results
                    try:
                        with open(results_file, 'r') as f:
                            previous_results = json.load(f)
                        # Create a map of video IDs to results
                        results_map = {r['video_info']['video_id']: r for r in previous_results if 'video_info' in r}
                        # Update with new results
                        for result in results:
                            if 'video_info' in result:
                                results_map[result['video_info']['video_id']] = result
                        results = list(results_map.values())
                    except Exception as e:
                        logging.warning(f"Error merging with previous results: {e}")
                
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Print summary
                success_count = sum(1 for r in results if r.get('success', False) or r.get('status') == 'skipped')
                logging.info(f"Successfully processed/skipped {success_count} out of {len(results)} videos")
                
                return results
                
            else:
                logging.warning("Falling back to channel scraping method")
                # Use existing Channel scraping logic
                return super().process_channel(channel_url, skip_existing)
                
        except Exception as e:
            logging.error(f"Error processing channel {channel_url}: {e}")
            raise

    def estimate_resources(self, video_infos: List[Dict]) -> Dict:
        """Estimate required disk space and processing time"""
        total_duration = sum(info['length'] for info in video_infos)
        
        # Rough estimates
        avg_video_size_per_minute = 5 * 1024 * 1024  # 5MB per minute
        avg_audio_size_per_minute = 1 * 1024 * 1024  # 1MB per minute
        avg_processing_time_per_minute = 15  # 15 seconds to process 1 minute
        
        estimated_video_size = total_duration * avg_video_size_per_minute / 60
        estimated_audio_size = total_duration * avg_audio_size_per_minute / 60
        estimated_time = total_duration * avg_processing_time_per_minute / 60
        
        return {
            'total_duration_hours': total_duration / 3600,
            'estimated_storage_gb': (estimated_video_size + estimated_audio_size) / (1024**3),
            'estimated_time_hours': estimated_time / 3600,
            'video_count': len(video_infos)
        }

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal gracefully"""
        if self._interrupt_received:
            logging.warning("Forced exit...")
            sys.exit(1)
        else:
            self._interrupt_received = True
            logging.warning("Interrupt received, finishing current tasks...")

def main():
    parser = argparse.ArgumentParser(description='Download and transcribe YouTube channel videos')
    parser.add_argument('channel_url', help='URL of the YouTube channel to process')
    parser.add_argument('--output-dir', default='channel_data',
                      help='Base directory for output files (default: channel_data)')
    parser.add_argument('--workers', type=int, default=2,
                      help='Number of parallel workers (default: 2)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                      help='Device to use for transcription (default: auto)')
    parser.add_argument('--model', default='turbo',
                      help='Whisper model to use (default: turbo)')
    parser.add_argument('--keep-video', action='store_true',
                      help='Keep video files after processing')
    parser.add_argument('--file-mode', choices=['skip', 'overwrite', 'rename'], default='skip',
                      help='How to handle existing files (default: skip)')
    parser.add_argument('--process-all', action='store_false', dest='skip_existing',
                      help='Process all videos, even if already transcribed')
    parser.add_argument('--estimate-only', action='store_true',
                      help='Only estimate required resources without processing')
    parser.add_argument('--max-videos', type=int,
                      help='Maximum number of videos to process')
    parser.add_argument('--min-duration', type=int,
                      help='Minimum video duration in seconds')
    parser.add_argument('--max-duration', type=int,
                      help='Maximum video duration in seconds')
    
    args = parser.parse_args()
    
    try:
        transcriber = ChannelTranscriber(
            output_base_dir=args.output_dir,
            max_workers=args.workers,
            device=args.device,
            file_mode=args.file_mode,
            whisper_model=args.model,
            keep_video=args.keep_video
        )
        
        # Get channel info and estimate resources
        channel = Channel(args.channel_url)
        videos = []
        
        logging.info(f"Fetching video information from channel: {channel.channel_name}")
        for url in tqdm(channel.video_urls[:args.max_videos], desc="Checking videos"):
            info = transcriber.get_video_info(url)
            if info and (
                (not args.min_duration or info['length'] >= args.min_duration) and
                (not args.max_duration or info['length'] <= args.max_duration)
            ):
                videos.append(info)
        
        estimates = transcriber.estimate_resources(videos)
        logging.info("\nResource Estimates:")
        logging.info(f"Videos to process: {estimates['video_count']}")
        logging.info(f"Total duration: {estimates['total_duration_hours']:.1f} hours")
        logging.info(f"Estimated storage needed: {estimates['estimated_storage_gb']:.1f} GB")
        logging.info(f"Estimated processing time: {estimates['estimated_time_hours']:.1f} hours")
        
        if args.estimate_only:
            return
        
        # Confirm if estimates are large
        if estimates['estimated_storage_gb'] > 10 or estimates['estimated_time_hours'] > 2:
            confirm = input("\nThis is a large processing job. Continue? (y/N): ")
            if confirm.lower() != 'y':
                logging.info("Aborted by user")
                return
        
        results = transcriber.process_channel(args.channel_url, skip_existing=args.skip_existing)
        
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 