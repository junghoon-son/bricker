from pathlib import Path
import whisper
import os
import ffmpeg
import json
import torch
from datetime import datetime
import time
import logging
from tqdm import tqdm
import argparse
import sys
import signal
from typing import Optional, List, Dict
import yt_dlp  # Replace pytubefix with yt-dlp
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
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
            
        # Load Whisper model with memory handling
        self.model = None  # Initialize later
        
        self._interrupt_received = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Initialize YouTube API
        load_dotenv()
        self.youtube_api = self._setup_youtube_api()
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': 'mp4',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',
            'ignoreerrors': True,
            'no_playlist_metadata': True,
        }
        
    def _setup_device(self, device: str) -> str:
        """Setup device with memory check"""
        if device == 'auto':
            if torch.cuda.is_available():
                try:
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB
                        logging.warning("GPU has limited memory, falling back to CPU")
                        return "cpu"
                    return "cuda"
                except Exception as e:
                    logging.warning(f"GPU check failed: {e}, falling back to CPU")
                    return "cpu"
            return "cpu"
        return device
    
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model with memory handling"""
        try:
            logging.info(f"Loading Whisper model '{self.whisper_model}' on {self.device}")
            
            if self.device == "cuda":
                # Clear CUDA cache before loading model
                torch.cuda.empty_cache()
                
            model = whisper.load_model(self.whisper_model)
            
            if self.device == "cuda":
                try:
                    model = model.to(self.device)
                except Exception as e:
                    logging.warning(f"Failed to move model to GPU: {e}, falling back to CPU")
                    self.device = "cpu"
                    torch.cuda.empty_cache()
                    
            return model.to(self.device)
            
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            sys.exit(1)
    
    def clean_filename(self, filename: str) -> str:
        return ''.join(c if c.isalnum() or c in '.-_ ' else '_' for c in filename)
    
    def get_video_info(self, url: str) -> Dict:
        """Get basic video information using YouTube API"""
        try:
            # Extract video ID from URL
            video_id = url.split('v=')[-1].split('&')[0]
    
            if not self.youtube_api:
                raise ValueError("YouTube API not initialized")
                
            # Get video details from API
            request = self.youtube_api.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                raise ValueError(f"No video found with ID {video_id}")
                
            video = response['items'][0]
            
            # Parse duration from ISO 8601 format (PT1H2M10S format)
            duration_str = video['contentDetails']['duration']
            
            # Initialize duration components
            hours = 0
            minutes = 0
            seconds = 0
            
            # Extract hours if present
            if 'H' in duration_str:
                hours_str = duration_str.split('PT')[-1].split('H')[0]
                hours = int(hours_str) if hours_str else 0
                duration_str = duration_str.split('H')[-1]
            else:
                duration_str = duration_str.split('PT')[-1]
                
            # Extract minutes if present
            if 'M' in duration_str:
                minutes_str = duration_str.split('M')[0]
                minutes = int(minutes_str) if minutes_str else 0
                duration_str = duration_str.split('M')[-1]
                
            # Extract seconds if present
            if 'S' in duration_str:
                seconds_str = duration_str.split('S')[0]
                seconds = int(seconds_str) if seconds_str else 0
                
            # Calculate total duration in seconds
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
            if total_seconds <= 0:
                logging.warning(f"Invalid duration for video {video_id}: {video['contentDetails']['duration']}")
                return None
                
            return {
                'title': video['snippet']['title'],
                'url': url,
                'video_id': video_id,
                'length': total_seconds,
                'publish_date': video['snippet']['publishedAt'],
                'views': video['statistics']['viewCount'],
                'author': video['snippet']['channelTitle'],
                'description': video['snippet']['description']
            }
        except Exception as e:
            logging.error(f"Error getting video info for {url}: {str(e)}")
            return None

    def get_existing_transcripts(self) -> Dict[str, Dict]:
        """Get a mapping of video IDs to existing transcript information"""
        existing = {}
        # Use glob pattern matching to find all transcript files at once
        for txt_file in self.transcripts_dir.glob("*.txt"):
            try:
                # Extract video_id from filename (assumes format: title_videoId.txt)
                video_id = txt_file.stem.split('_')[-1]
                
                # Only check json file if txt exists since that's faster
                json_file = self.transcripts_dir / f"{txt_file.stem}.json"
                
                if json_file.exists():
                    existing[video_id] = {
                        'json_path': json_file,
                        'txt_path': txt_file,
                        'processed_at': datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
                    }
            except Exception as e:
                logging.warning(f"Error reading existing transcript {txt_file}: {e}")
        return existing

    def process_video(self, video_url: str) -> Dict:
        """Process a single video: download, convert to audio, and transcribe"""
        video_path = None
        audio_path = None
        
        try:
            # Initialize model if not already done
            if self.model is None:
                self.model = self._load_whisper_model()
            
            # Quick check for video ID first
            video_id = video_url.split('v=')[-1].split('&')[0]
            if not video_id:
                raise ValueError("Could not extract video ID from URL")

            # Get basic info without downloading
            info = self.get_video_info(video_url)
            if not info:
                raise ValueError(f"Could not get video info for {video_url}")
            
            clean_title = self.clean_filename(info['title'])
            
            # Set up file paths
            video_path = self.videos_dir / f"{clean_title}_{video_id}.mp4"
            audio_path = self.audio_dir / f"{clean_title}_{video_id}.mp3"
            transcript_json = self.transcripts_dir / f"{clean_title}_{video_id}.json"
            transcript_txt = self.transcripts_dir / f"{clean_title}_{video_id}.txt"
            
            # Quick check if already processed
            if self.file_mode == 'skip' and transcript_txt.exists() and transcript_json.exists():
                logging.info(f"Already processed video: {clean_title}")
                return {
                    'title': info['title'],
                    'url': video_url,
                    'success': True,
                    'status': 'skipped'
                }
            
            # Download video with retries
            logging.info(f"Downloading: {clean_title}")
            download_opts = {
                **self.ydl_opts,
                'outtmpl': str(video_path),
                'quiet': True,
                'format': 'best[ext=mp4]/best',  # More flexible format selection
                'ignoreerrors': True,
                'retries': 3,  # Add retries
                'fragment_retries': 3,
            }
            
            try:
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    download_info = ydl.download([video_url])
                    if download_info != 0 or not video_path.exists():
                        raise ValueError("Download failed")
            except Exception as e:
                logging.error(f"Download failed for {clean_title}: {e}")
                # Clean up any partial downloads
                if video_path.exists():
                    video_path.unlink()
                raise
            
            try:
                # Convert to audio with better error handling
                logging.info(f"Converting to audio: {clean_title}")
                
                # Add more ffmpeg options for better compatibility
                input_stream = ffmpeg.input(str(video_path))
                output_stream = ffmpeg.output(
                    input_stream, 
                    str(audio_path),
                    acodec='libmp3lame',
                    q=0,
                    ar='44100',  # Set standard audio sample rate
                    ac=2,        # Set stereo audio
                    loglevel='warning'  # Increase ffmpeg logging
                )
                
                try:
                    # Capture ffmpeg output for debugging
                    stdout, stderr = ffmpeg.run(
                        output_stream, 
                        capture_stdout=True, 
                        capture_stderr=True,
                        overwrite_output=True
                    )
                except ffmpeg.Error as e:
                    # Log detailed ffmpeg error
                    logging.error(f"FFmpeg error for {clean_title}:")
                    if e.stderr:
                        logging.error(f"FFmpeg stderr: {e.stderr.decode()}")
                    if e.stdout:
                        logging.error(f"FFmpeg stdout: {e.stdout.decode()}")
                    raise ValueError(f"Audio conversion failed: {str(e)}")
                
                # Verify audio file was created and has size
                if not audio_path.exists():
                    raise ValueError("Audio file was not created")
                
                if audio_path.stat().st_size == 0:
                    raise ValueError("Audio file is empty")
                    
                # Transcribe with better memory handling
                logging.info(f"Transcribing: {clean_title}")
                try:
                    # Clear memory before transcription
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    else:
                        import gc
                        gc.collect()
                        
                    # Load audio in chunks if it's large
                    audio_size = audio_path.stat().st_size / (1024 * 1024)  # Size in MB
                    if audio_size > 100:  # If audio is larger than 100MB
                        logging.info(f"Large audio file ({audio_size:.1f}MB), using chunked processing")
                        # Force CPU for large files to avoid GPU memory issues
                        original_device = self.device
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                        torch.cuda.empty_cache()
                    
                    # Transcribe with error handling
                    try:
                        result = self.model.transcribe(
                            str(audio_path),
                            fp16=False,  # Disable fp16 to avoid some memory issues
                            language='en'  # Set language if known
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.warning("Memory error during transcription, trying with smaller model")
                            # Try falling back to smaller model
                            if self.whisper_model not in ['tiny', 'base']:
                                original_model = self.whisper_model
                                self.whisper_model = 'base'
                                self.model = self._load_whisper_model()
                                result = self.model.transcribe(str(audio_path))
                                # Restore original model
                                self.whisper_model = original_model
                                self.model = self._load_whisper_model()
                        else:
                            raise  # Re-raise the RuntimeError if it's not a memory issue
                
                except Exception as e:
                    logging.error(f"Transcription failed: {e}")
                    raise ValueError(f"Transcription failed: {str(e)}")
                
                # Save transcripts
                with open(transcript_json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                with open(transcript_txt, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {info['title']}\n")
                    f.write(f"URL: {video_url}\n")
                    f.write(f"Author: {info['author']}\n")
                    f.write(f"Language: {result['language']}\n")
                    f.write("\nTranscription:\n")
                    f.write(result['text'])
                    
                    f.write("\n\nDetailed segments with timestamps:\n")
                    for segment in result['segments']:
                        start_time = int(segment['start'])
                        end_time = int(segment['end'])
                        text = segment['text']
                        f.write(f"\n[{start_time//60}:{start_time%60:02d} - {end_time//60}:{end_time%60:02d}] {text}")
                
                return {
                    'title': info['title'],
                    'url': video_url,
                    'success': True
                }
                
            except Exception as e:
                logging.error(f"Error processing {clean_title}: {str(e)}")
                # Clean up files on error
                if video_path and video_path.exists():
                    video_path.unlink()
                if audio_path and audio_path.exists():
                    audio_path.unlink()
                raise
                
        except Exception as e:
            error_msg = f"Error processing video {video_url}: {str(e)}"
            logging.error(error_msg)
            return {
                'url': video_url,
                'error': error_msg,
                'success': False
            }
        finally:
            # Always try to clean up temporary files
            try:
                if not self.keep_video and video_path and video_path.exists():
                    video_path.unlink()
                if audio_path and audio_path.exists():
                    audio_path.unlink()
            except Exception as cleanup_error:
                logging.warning(f"Error during cleanup: {cleanup_error}")

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
            # Get total video count first with a minimal request
            initial_request = self.youtube_api.search().list(
                part="id",
                channelId=channel_url,
                maxResults=1,
                type="video"
            )
            initial_response = initial_request.execute()
            total_videos = initial_response.get('pageInfo', {}).get('totalResults', 0)
            
            # Quick check against existing transcripts
            existing_count = len(self.get_existing_transcripts())
            completion_ratio = existing_count / total_videos if total_videos > 0 else 0
            
            logging.info(f"Found {existing_count} transcripts out of {total_videos} videos ({completion_ratio:.1%} complete)")
            
            # If we've processed most videos, only check recent ones
            if completion_ratio > 0.3:  # Over 30% complete
                logging.info("Channel partially processed, checking only recent videos...")
                max_results = min(max_results or 50, 50)  # Check at most 50 recent videos
            
            # Continue with regular video fetching
            request = self.youtube_api.search().list(
                part="snippet",
                channelId=channel_url,
                maxResults=max_results or 50,
                order="date",  # Get newest first
                type="video"
            )
            
            response = request.execute()
            
            if not response.get('items'):
                return []
                
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
            videos_data = []
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

            return videos_data
            
        except Exception as e:
            logging.error(f"Error fetching channel videos: {e}")
            return None

    def process_channel(self, channel_url: str, skip_existing: bool = True) -> List[Dict]:
        """Process all videos from a YouTube channel"""
        try:
            # Get existing transcripts first - this is fast
            existing_transcripts = self.get_existing_transcripts() if skip_existing else {}
            
            # Try to get videos using API first
            videos_data = self.get_channel_videos(channel_url)
            
            if videos_data:
                # Pre-filter videos that are already processed
                videos_to_process = []
                skipped_videos = []
                
                total_duration = 0
                processed_duration = 0
                
                # Use a set for faster lookups
                existing_ids = set(existing_transcripts.keys())
                
                for video in videos_data:
                    try:
                        video_id = video['video_id']
                        duration = video.get('length', 0)
                        
                        # Fast lookup using set membership
                        if skip_existing and video_id in existing_ids:
                            processed_duration += duration
                            skipped_videos.append({
                                'video_info': video,
                                'transcript_info': existing_transcripts[video_id],
                                'status': 'skipped'
                            })
                        else:
                            videos_to_process.append((video['url'], video))
                            
                        total_duration += duration
                            
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
        # Filter out videos with None duration and convert string durations if needed
        valid_videos = [info for info in video_infos if info and info.get('length') is not None]
        
        if not valid_videos:
            return {
                'total_duration_hours': 0,
                'estimated_storage_gb': 0,
                'estimated_time_hours': 0,
                'video_count': 0
            }
        
        total_duration = sum(info['length'] for info in valid_videos)
        
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
            'video_count': len(valid_videos)
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
        
        # Configure yt-dlp options for channel extraction
        channel_opts = {
            'format': 'best',
            'extract_flat': True,
            'quiet': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'playlistreverse': False,  # Get newest videos first
        }
        
        if args.max_videos:
            channel_opts['playlistend'] = args.max_videos
            
        # Try different URL formats for the channel
        channel_url = args.channel_url
        if '@' in channel_url:
            # Handle @username format
            username = channel_url.split('@')[-1].split('/')[0]
            channel_url = f"https://www.youtube.com/@{username}/videos"
        elif 'channel/' in channel_url:
            # Handle channel ID format
            channel_id = channel_url.split('channel/')[-1].split('/')[0]
            channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
        elif 'user/' in channel_url:
            # Handle legacy username format
            username = channel_url.split('user/')[-1].split('/')[0]
            channel_url = f"https://www.youtube.com/user/{username}/videos"
            
        logging.info(f"Fetching videos from: {channel_url}")
        
        # Get channel videos
        with yt_dlp.YoutubeDL(channel_opts) as ydl:
            try:
                channel_info = ydl.extract_info(channel_url, download=False)
                if not channel_info:
                    raise ValueError("Could not get channel info")
                    
                videos = channel_info.get('entries', [])
                if not videos:
                    raise ValueError("No videos found in channel")
                    
                # Get valid video URLs
                video_urls = []
                for video in videos:
                    if video and isinstance(video, dict) and 'id' in video and video['id']:
                        video_urls.append(f"https://www.youtube.com/watch?v={video['id']}")
                    else:
                        logging.warning(f"Skipping invalid video entry: {video}")
                        
            except Exception as e:
                logging.error(f"Error extracting channel info: {e}")
                return
                
        if not video_urls:
            logging.error("No valid videos found in channel")
            return
            
        logging.info(f"Found {len(video_urls)} videos in channel")
        
        # Process videos
        results = []
        for url in tqdm(video_urls, desc="Processing videos"):
            try:
                result = transcriber.process_video(url)
                if result:
                    results.append(result)
                time.sleep(1)  # Add small delay between videos
            except Exception as e:
                logging.error(f"Error processing {url}: {e}")
                continue
                
        # Print summary
        success_count = sum(1 for r in results if r.get('success', False))
        logging.info(f"\nSuccessfully processed {success_count} out of {len(video_urls)} videos")
        
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 