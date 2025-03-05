#!/usr/bin/env python3
# YouTube Music Playlist Downloader
version = "1.4.1"

import os
import re
import sys
import copy
import json
import time
import logging
import requests
import subprocess
import concurrent.futures
from PIL import Image
from io import BytesIO
from pathlib import Path
from langcodes import Language
from yt_dlp import YoutubeDL, postprocessor
from urllib.parse import urlparse, parse_qs
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TRCK, TALB, TDRC, WOAR, SYLT, USLT, error
from datetime import datetime

# Setup logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration with file and console handlers."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_date_format = '%Y-%m-%d %H:%M:%S'
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"youtube_music_downloader_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, log_date_format))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, log_date_format))
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return root_logger

logger = setup_logging()

class FilePathCollector(postprocessor.common.PostProcessor):
    def __init__(self):
        super().__init__(None)
        self.file_paths = []

    def run(self, information):
        self.file_paths.append(information['filepath'])
        return [], information

class SongFileInfo:
    def __init__(self, video_id, name, file_name, file_path, track_num):
        self.video_id = video_id
        self.name = name
        self.file_name = file_name
        self.file_path = file_path
        self.track_num = track_num

def write_config(file_path: Path, config: dict):
    """Write configuration to a file with error handling."""
    try:
        with file_path.open("w") as f:
            json.dump(config, f, indent=4)
    except IOError as e:
        logger.error(f"Failed to write config to {file_path}: {e}")

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "ffmpeg not found. Please install ffmpeg and add it to your PATH.\n"
            "Download: https://www.ffmpeg.org/download.html"
        )
        return False

def get_playlist_info(config: dict):
    """Retrieve playlist information with retry logic."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "force_generic_extractor": True
    }
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(config["url"], download=False)
        except Exception as e:
            if "getaddrinfo failed" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Network error, retrying in {retry_delay}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to get playlist info: {e}")

def convert_image_type(image, image_type):
    """Convert image to specified type."""
    with BytesIO() as f:
        image.convert("RGB").save(f, format=image_type)
        return f.getvalue()

def update_track_num(file_path: Path, track_num: int):
    """Update track number in file metadata."""
    try:
        tags = ID3(str(file_path))
        tags.add(TRCK(encoding=3, text=str(track_num)))
        tags.save(v2_version=3)
    except error as e:
        logger.warning(f"Failed to update track number for {file_path}: {e}")

def update_file_order(playlist_dir: Path, song_file_info, track_num: int, config: dict, missing_video: bool):
    """Update file order and name based on track number."""
    file_name = song_file_info.file_name
    if config["track_num_in_name"]:
        file_name = re.sub(r"^[0-9]+\. ", "", file_name)
        file_name = f"{track_num}. {file_name}"
    new_path = playlist_dir / file_name

    if song_file_info.track_num != track_num and config["include_metadata"]["track"]:
        action = "due to missing video" if missing_video else ""
        logger.info(f"Reordering '{song_file_info.name}' from {song_file_info.track_num} to {track_num} {action}")
        update_track_num(Path(song_file_info.file_path), track_num)

    if song_file_info.file_path != str(new_path):
        logger.info(f"Renaming '{song_file_info.file_name}' to '{file_name}'")
        try:
            os.rename(song_file_info.file_path, new_path)
        except OSError as e:
            logger.error(f"Failed to rename file: {e}")
    return new_path

def get_metadata_map():
    """Return mapping of metadata types to ID3 tags."""
    return {
        "title": ["TIT2"],
        "cover": ["APIC:Front cover"],
        "track": ["TRCK"],
        "artist": ["TPE1"],
        "album": ["TALB"],
        "date": ["TDRC"],
        "url": ["WOAR"],
        "lyrics": ["SYLT", "USLT"]
    }

def flatten(l):
    """Flatten a nested list."""
    return [item for sublist in l for item in sublist]

def get_metadata_dict(tags):
    """Get dictionary of metadata from tags."""
    return {tag: tags.getall(tag) for tag in flatten(get_metadata_map().values())}

def valid_metadata(config: dict, metadata_dict: dict):
    """Check if required metadata is present."""
    include_metadata = config["include_metadata"].copy()
    include_metadata["url"] = True  # WOAR is always required
    selected_tags = flatten([v for k, v in get_metadata_map().items() if include_metadata[k]])
    return all(metadata_dict.get(tag) for tag in selected_tags)

def get_song_info_ytdl(track_num: int, config: dict):
    """Configure YoutubeDL for song info extraction."""
    name_format = config["name_format"]
    if config["track_num_in_name"]:
        name_format = f"{track_num}. {name_format}"
    ytdl_opts = {
        "quiet": True,
        "geo_bypass": True,
        "outtmpl": name_format,
        "format": config["audio_format"],
        "cookiefile": config["cookie_file"] or None,
        "cookiesfrombrowser": tuple(config["cookies_from_browser"].split(":")) if config["cookies_from_browser"] else None,
        "writesubtitles": True,
        "allsubtitles": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": config["audio_codec"],
            "preferredquality": config["audio_quality"],
        }]
    }
    return YoutubeDL(ytdl_opts)

def get_song_info(track_num: int, link: str, config: dict):
    """Retrieve song metadata with retry logic."""
    ydl = get_song_info_ytdl(track_num, config)
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            return ydl.extract_info(link, download=False)
        except Exception as e:
            if "getaddrinfo failed" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Network error, retrying in {retry_delay}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to get song info: {e}")

def get_subtitles_url(subtitles, lang):
    """Get URL for JSON3 subtitles in specified language."""
    return next((sub["url"] for sub in subtitles.get(lang, []) if sub["ext"] == "json3"), None)

def generate_metadata(file_path: Path, link: str, track_num: int, playlist_name: str, config: dict, regenerate: bool, force_update: bool):
    """Generate or update song metadata."""
    try:
        tags = ID3(str(file_path))
    except error:
        if force_update:
            try:
                info_dict = get_song_info(track_num, link, config)
                info_dict["ext"] = config["audio_codec"]
                return get_song_info_ytdl(track_num, config).prepare_filename(info_dict)
            except Exception as e:
                logger.error(f"Cannot update filename for {file_path}: {e}")
        return ""

    metadata_dict = get_metadata_dict(tags)
    force_update_file_name = ""
    if force_update:
        for tag in list(metadata_dict.keys()):
            if tag != "WOAR":
                tags.delall(tag)
        metadata_dict = get_metadata_dict(tags)

    if regenerate or force_update or not valid_metadata(config, metadata_dict):
        try:
            info_dict = get_song_info(track_num, link, config)
            if force_update:
                info_dict["ext"] = config["audio_codec"]
                force_update_file_name = get_song_info_ytdl(track_num, config).prepare_filename(info_dict)

            title = info_dict.get("title", "Unknown Title")
            logger.info(f"Updating metadata for '{title}'...")
            include_metadata = config["include_metadata"]

            if include_metadata["cover"]:
                try:
                    thumbnail = info_dict.get("thumbnail")
                    if thumbnail:
                        resp = requests.get(thumbnail, stream=True, timeout=10)
                        resp.raise_for_status()
                        img = Image.open(resp.raw)
                        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                        width, height = img.size
                        min_dim = min(width, height)
                        left = (width - min_dim) // 2
                        top = (height - min_dim) // 2
                        img = img.crop((left, top, left + min_dim, top + min_dim))
                        img_data = convert_image_type(img, config["image_format"])
                        tags.add(APIC(3, f"image/{config['image_format']}", 3, "Front cover", img_data))
                except Exception as e:
                    logger.warning(f"Failed to update thumbnail: {e}")

            if include_metadata["track"]:
                tags.add(TRCK(encoding=3, text=str(track_num)))
            if include_metadata["date"]:
                upload_date = info_dict.get("upload_date")
                if upload_date:
                    tags.add(TDRC(encoding=3, text=time.strftime('%Y-%m-%d', time.strptime(upload_date, '%Y%m%d'))))
            tags.add(WOAR(link))

            if include_metadata["lyrics"]:
                try:
                    subtitles = info_dict.get("subtitles", {})
                    req_subtitles = info_dict.get("requested_subtitles", {})
                    if req_subtitles:
                        lang = "en"
                        lyrics_langs = config["lyrics_langs"]
                        strict = config["strict_lang_match"]
                        subtitles_url = None
                        if not lyrics_langs:
                            lang = "en" if "en" in req_subtitles else next(iter(req_subtitles), "en")
                            subtitles_url = get_subtitles_url(subtitles, lang)
                        else:
                            for l in lyrics_langs:
                                if l in req_subtitles:
                                    lang = l
                                    subtitles_url = get_subtitles_url(subtitles, lang)
                                    break
                            if not subtitles_url and not strict:
                                lang = next(iter(req_subtitles))
                                subtitles_url = get_subtitles_url(subtitles, lang)

                        if subtitles_url:
                            resp = requests.get(subtitles_url, timeout=10)
                            content = json.loads(resp.text)
                            synced_lyrics = []
                            unsynced_lyrics = []
                            last_time = -1
                            current_line = ""
                            current_time = 0
                            for event in content.get("events", []):
                                timestamp = event.get("tStartMs", 0)
                                line = "".join(seg["utf8"] for seg in event.get("segs", [])).strip()
                                if not line or (timestamp - last_time < 1000 and line in unsynced_lyrics[-1:]):
                                    continue
                                if timestamp == last_time:
                                    current_line += "\n" + line
                                else:
                                    if current_line:
                                        synced_lyrics.append((current_line, current_time))
                                        unsynced_lyrics.append(current_line)
                                    current_line = line
                                    current_time = timestamp
                                last_time = timestamp
                            if current_line:
                                synced_lyrics.append((current_line, current_time))
                                unsynced_lyrics.append(current_line)

                            lang = Language.get(lang).to_alpha3() if lang in Language else "eng"
                            if synced_lyrics:
                                tags.add(SYLT(encoding=3, lang=lang, format=2, type=1, text=synced_lyrics))
                            tags.add(USLT(encoding=3, lang=lang, text="\n".join(unsynced_lyrics) or "Lyrics unavailable"))
                except Exception as e:
                    logger.warning(f"Failed to update lyrics: {e}")

            if include_metadata["title"]:
                track = info_dict.get("track")
                tags.add(TIT2(encoding=3, text=title if config["use_title"] or not track else track))
            if include_metadata["artist"]:
                artist = info_dict.get("artist")
                uploader = info_dict.get("uploader", "Unknown Artist")
                tags.add(TPE1(encoding=3, text=uploader if config["use_uploader"] or not artist else artist))
            if include_metadata["album"]:
                album = info_dict.get("album")
                tags.add(TALB(encoding=3, text=playlist_name if config["use_playlist_name"] else (album or "Unknown Album")))

            tags.save(v2_version=3)
        except Exception as e:
            logger.error(f"Failed to update metadata for {file_path}: {e}")
    return force_update_file_name

def download_song(link: str, playlist_dir: Path, track_num: int, config: dict):
    """Download a song with retry logic and progress tracking."""
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            playlist_dir.mkdir(parents=True, exist_ok=True)
            name_format = config["name_format"]
            if config["track_num_in_name"]:
                name_format = f"{track_num}. {name_format}"
            ytdl_opts = {
                "outtmpl": str(playlist_dir / name_format),
                "ignoreerrors": True,
                "format": config["audio_format"],
                "cookiefile": config["cookie_file"] or None,
                "cookiesfrombrowser": tuple(config["cookies_from_browser"].split(":")) if config["cookies_from_browser"] else None,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": config["audio_codec"],
                    "preferredquality": config["audio_quality"],
                }],
                "geo_bypass": True,
                "quiet": not config["verbose"],
                "external_downloader_args": ["-loglevel", "panic"] if not config["verbose"] else [],
                "progress_hooks": [lambda d: logger.debug(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']}")] if config["verbose"] else []
            }
            with YoutubeDL(ytdl_opts) as ydl:
                collector = FilePathCollector()
                ydl.add_post_processor(collector)
                ydl.download([link])
                if not collector.file_paths:
                    raise Exception("Download failed, video may be unavailable")
                logger.info(f"Successfully downloaded: {collector.file_paths[0]}")
                return 0, collector.file_paths[0]
        except Exception as e:
            if "getaddrinfo failed" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Network error, retrying in {retry_delay}s (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Download failed for {link}: {e}")
                return 1, None

def download_song_and_update(video_info, playlist, link: str, playlist_dir: Path, track_num: int, config: dict):
    """Download a song and update its metadata."""
    try:
        if video_info.get("availability") == "unavailable":
            reason = video_info.get("availability_reason", "unknown reason")
            raise Exception(f"Video unavailable: {video_info.get('title', 'Unknown Title')} ({reason})")
        result, file_path = download_song(link, playlist_dir, track_num, config)
        if result != 0 or not file_path:
            if not video_info.get("channel_id"):
                raise Exception(f"Video unavailable: {video_info.get('title', 'Unknown Title')}")
        if file_path:
            generate_metadata(Path(file_path), link, track_num, playlist["title"], config, False, False)
        return None, track_num
    except Exception as e:
        return f"Failed to download #{track_num} '{link}': {e}", track_num

def update_song(video_info, song_file_info, file_path: Path, link: str, track_num: int, playlist_name: str, config: dict, regenerate: bool, force_update: bool):
    """Update song metadata and handle unavailable videos."""
    video_unavailable = False
    errors = []
    try:
        if video_info.get("availability") == "unavailable":
            reason = video_info.get("availability_reason", "unknown reason")
            raise Exception(f"Video unavailable: {video_info.get('title', 'Unknown Title')} ({reason})")
        new_name = generate_metadata(file_path, link, track_num, playlist_name, config, regenerate, force_update)
        if force_update and new_name:
            new_path = Path(playlist_name) / new_name
            if str(file_path) != str(new_path):
                logger.info(f"Renaming '{file_path.stem}' to '{new_path.stem}'")
                os.rename(file_path, new_path)
    except Exception as e:
        errors.append(f"Failed to update metadata for #{track_num} '{link}': {e}")
        video_unavailable = "unavailable" in str(e).lower()

    if video_info.get("channel_id") is None or video_unavailable:
        msg = f"Song '{song_file_info.name}' is unavailable but exists locally"
        if not video_unavailable and video_info.get("title"):
            msg += f" - {video_info['title']}"
        errors.append(msg)
    return "\n".join(errors) if errors else None

def format_file_name(file_name: str):
    """Sanitize file name for safe use."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    return sanitized.strip(". ") or "unnamed_playlist"

def get_url_parameter(url: str, param: str):
    """Extract URL parameter."""
    return parse_qs(urlparse(url).query).get(param, [""])[0]

def get_video_id_from_metadata(tags):
    """Extract video ID from WOAR tag."""
    links = tags.getall("WOAR")
    if not links or len(links) > 1:
        raise Exception("WOAR tag is invalid")
    return get_url_parameter(str(links[0]), "v")

def get_song_file_info(playlist_dir: Path, file_name: str):
    """Retrieve song file information."""
    file_path = playlist_dir / file_name
    try:
        tags = ID3(str(file_path))
        video_id = get_video_id_from_metadata(tags)
        name = str(tags.get("TIT2", file_name))
        track_num = int(str(tags.get("TRCK", 0)))
        return SongFileInfo(video_id, name, file_name, str(file_path), track_num)
    except Exception as e:
        logger.warning(f"Ignoring invalid song file '{file_name}': {e}")
        return None

def get_song_file_infos(playlist_dir: Path):
    """Get information for all song files in directory."""
    song_file_infos = {}
    duplicates = {}
    for file_name in os.listdir(playlist_dir):
        info = get_song_file_info(playlist_dir, file_name)
        if info:
            if info.video_id in song_file_infos:
                duplicates.setdefault(info.video_id, [song_file_infos[info.video_id].file_name]).append(info.file_name)
            else:
                song_file_infos[info.video_id] = info
    if duplicates:
        raise Exception("\n".join([f"Duplicate files for video '{vid}': {', '.join(files)}" for vid, files in duplicates.items()]))
    return song_file_infos

def setup_include_metadata_config():
    """Setup default metadata inclusion config."""
    return {key: True for key in get_metadata_map() if key != "url"}

def copy_config(src: dict, dst: dict):
    """Copy configuration values recursively."""
    for key, value in dst.items():
        if isinstance(value, dict):
            if key in src and isinstance(src[key], dict):
                copy_config(src[key], value)
        elif key in src and type(src[key]) == type(value):
            dst[key] = src[key]

def get_override_config(video_id: str, base_config: dict):
    """Get config with overrides for a specific video."""
    config = copy.deepcopy(base_config)
    if video_id in base_config.get("overrides", {}):
        copy_config(base_config["overrides"][video_id], config)
    return config

def validate_config(config: dict):
    """Validate and set default configuration."""
    defaults = {
        "url": "",
        "reverse_playlist": False,
        "use_title": True,
        "use_uploader": True,
        "use_playlist_name": True,
        "sync_folder_name": True,
        "use_threading": True,
        "thread_count": 0,
        "retain_missing_order": False,
        "name_format": "%(title)s-%(id)s.%(ext)s",
        "track_num_in_name": True,
        "audio_format": "bestaudio/best",
        "audio_codec": "mp3",
        "audio_quality": "5",
        "image_format": "jpeg",
        "lyrics_langs": [],
        "strict_lang_match": False,
        "cookie_file": "",
        "cookies_from_browser": "",
        "verbose": False,
        "include_metadata": setup_include_metadata_config(),
        "overrides": {}
    }
    for key, default in defaults.items():
        config.setdefault(key, default)
        if isinstance(default, dict):
            copy_config(config[key], defaults[key])
    return config

def generate_playlist(base_config: dict, config_file_name: str, update: bool, force_update: bool, regenerate: bool, single_playlist: bool, current_playlist_name: str = None, track_num_to_update: int = None):
    """Generate or update a playlist."""
    config = validate_config(base_config)
    playlist = get_playlist_info(config)
    if "entries" not in playlist:
        raise Exception("No videos found in playlist")
    entries = playlist["entries"]

    playlist_name = "." if single_playlist else format_file_name(playlist["title"])
    playlist_dir = Path(playlist_name)
    if not single_playlist and update and current_playlist_name and current_playlist_name != playlist_name and config["sync_folder_name"]:
        try:
            old_dir = Path(current_playlist_name)
            if old_dir.exists():
                new_dir = Path(playlist_name)
                if new_dir.exists():
                    raise FileExistsError(f"Directory '{new_dir}' already exists")
                old_dir.rename(new_dir)
                logger.info(f"Renamed playlist from '{current_playlist_name}' to '{playlist_name}'")
            else:
                playlist_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created new directory '{playlist_name}'")
        except Exception as e:
            logger.error(f"Failed to handle directory: {e}")
            raise
    else:
        playlist_dir.mkdir(parents=True, exist_ok=True)

    config_path = playlist_dir / config_file_name
    write_config(config_path, config)
    song_file_infos = get_song_file_infos(playlist_dir)

    track_num = 1
    skipped_videos = 0
    updated_video_ids = []
    for video_id, info in song_file_infos.items():
        if config["retain_missing_order"] and not any(e and e["id"] == video_id for e in entries):
            index = info.track_num - 1
            if index >= len(entries):
                entries.extend([None] * (index - len(entries) + 1))
            entries.insert(index, {"id": video_id, "channel_id": None, "title": None})

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config["thread_count"] or None) if config["use_threading"] else None
    futures = []

    for i, video_info in enumerate(entries):
        if not video_info:
            continue
        track_num = i + 1 - skipped_videos
        video_id = video_info["id"]
        link = f"https://www.youtube.com/watch?v={video_id}"
        song_info = song_file_infos.get(video_id)
        if track_num_to_update and (not song_info or song_info.track_num != track_num):
            continue

        cfg = get_override_config(video_id, config)
        updated_video_ids.append(video_id)

        if track_num_to_update:
            if song_info:
                file_path = Path(song_info.file_path)
                try:
                    new_name = generate_metadata(file_path, link, song_info.track_num, playlist["title"], cfg, regenerate, True)
                    if new_name:
                        new_path = playlist_dir / new_name
                        os.rename(file_path, new_path)
                except Exception as e:
                    logger.error(f"Failed to update song #{track_num_to_update}: {e}")
            else:
                logger.error(f"Song #{track_num_to_update} not downloaded yet")
            return

        if not song_info:
            logger.info(f"Downloading '{link}' ({track_num}/{len(entries) - skipped_videos})")
            task = executor.submit(download_song_and_update, video_info, playlist, link, playlist_dir, track_num, cfg) if executor else download_song_and_update(video_info, playlist, link, playlist_dir, track_num, cfg)
            futures.append((task, track_num)) if executor else (lambda e, t: logger.error(e) or (skipped_videos + 1) if e else skipped_videos)(*task)
        else:
            logger.info(f"Skipped '{link}' ({track_num}/{len(entries) - skipped_videos})")
            file_path = update_file_order(playlist_dir, song_info, track_num, cfg, False) if not executor else song_info.file_path
            task = executor.submit(update_song, video_info, song_info, Path(file_path), link, track_num, playlist["title"], cfg, regenerate, force_update) if executor else update_song(video_info, song_info, Path(file_path), link, track_num, playlist["title"], cfg, regenerate, force_update)
            futures.append((task, track_num)) if executor else logger.error(task) if task else None

    if executor:
        skipped_videos = sum(1 for f, t in futures if isinstance(f.result(), tuple) and f.result()[0])
        temp_infos = get_song_file_infos(playlist_dir)
        for i, video_info in enumerate(entries):
            if not video_info or (i + 1) in [t for f, t in futures if isinstance(f.result(), tuple) and f.result()[0]]:
                skipped_videos += 1 if not video_info else 0
                continue
            track_num = i + 1 - skipped_videos
            video_id = video_info["id"]
            if video_id in temp_infos:
                cfg = get_override_config(video_id, config)
                update_file_order(playlist_dir, temp_infos[video_id], track_num, cfg, False)
        executor.shutdown(wait=False)

    if track_num_to_update:
        logger.error(f"Song #{track_num_to_update} not found or unavailable")
        return

    track_num = len(entries) - skipped_videos + 1
    for video_id, info in song_file_infos.items():
        if video_id not in updated_video_ids:
            cfg = get_override_config(video_id, config)
            update_file_order(playlist_dir, info, track_num, cfg, True)
            track_num += 1

    logger.info("Download finished.")

def get_existing_playlists(directory: Path, config_file_name: str):
    """Get list of existing playlists."""
    playlists = []
    playlist_ids = {}
    for subdir in directory.iterdir():
        if subdir.is_dir() and (subdir / config_file_name).exists():
            try:
                with (subdir / config_file_name).open() as f:
                    cfg = json.load(f)
                pid = get_url_parameter(cfg["url"], "list")
                if pid in playlist_ids:
                    raise FileExistsError(f"Duplicate playlist ID '{pid}' in {playlist_ids[pid]} and {subdir}")
                playlists.append({
                    "playlist_name": subdir.name,
                    "config_file": str(subdir / config_file_name),
                    "last_updated": time.strftime('%x %X', time.localtime((subdir / config_file_name).stat().st_mtime))
                })
                playlist_ids[pid] = subdir.name
            except json.JSONDecodeError as e:
                logger.error(f"Invalid config in {subdir / config_file_name}: {e}")
    return playlists

def get_bool_option_response(prompt: str, default: bool):
    """Get boolean response from user."""
    choice = "Y/n" if default else "y/N"
    while True:
        resp = input(f"{prompt} ({choice}): ").lower()
        if resp in ("y", "") and default:
            return True
        if resp == "n" or (resp == "" and not default):
            return False
        print("Invalid response, please type 'y' or 'n'.")

def get_index_option_response(prompt: str, count: int, allow_all: bool = False):
    """Get index response from user."""
    while True:
        resp = input(f"{prompt} (1 to {count}{' or all' if allow_all else ''}): ").lower()
        if allow_all and resp == "all":
            return "all"
        try:
            idx = int(resp) - 1
            if 0 <= idx < count:
                return idx
        except ValueError:
            pass
        print("Invalid response, please enter a valid number.")

def get_numeric_option_response(prompt: str):
    """Get numeric response from user."""
    while True:
        try:
            num = int(input(f"{prompt}: "))
            if num > 0:
                return num
        except ValueError:
            print("Invalid response, please enter a number > 0.")

if __name__ == "__main__":
    logger.info(
        f"YouTube Music Playlist Downloader v{version}\n"
        "Downloads and updates YouTube playlists as local music albums.\n"
        "- Stores songs in folders named by playlist title\n"
        "- Updates existing albums with new/missing songs\n"
        "- Generates metadata (title, artist, album, lyrics, etc.)\n"
        "- Embeds video thumbnails as cover art\n"
        "Note: Antivirus may block this program or ffmpeg; add exclusions if needed."
    )

    config_file_name = ".playlist_config.json"
    single_playlist = Path(config_file_name).exists()
    if single_playlist:
        logger.info(f"Running in single playlist mode due to '{config_file_name}' in current directory.")

    options = {
        "download": "Download a playlist from YouTube",
        "generate": "Generate default playlist config",
        "change": "Change current working directory",
        "exit": "Exit"
    }

    while True:
        try:
            if not check_ffmpeg():
                continue
            config = {}
            playlists_data = get_existing_playlists(Path("."), config_file_name) if not single_playlist else []
            if not single_playlist and playlists_data:
                options.update({
                    "update": "Update previously saved playlist",
                    "song": "Update a single song in playlist",
                    "modify": "Modify previously saved playlist"
                })

            if single_playlist:
                with Path(config_file_name).open() as f:
                    config = json.load(f)
                current_playlist_name = os.getcwd()
            else:
                print("\n".join(f"{i+1}. {v}" for i, v in enumerate(options.values())) + "\n")
                choice = list(options.values())[get_index_option_response("Select an option", len(options))]
                action = next(k for k, v in options.items() if v == choice)

                if action == "download":
                    config["url"] = input("Enter playlist URL: ")
                    for p in playlists_data:
                        with Path(p["config_file"]).open() as f:
                            cfg = json.load(f)
                        if get_url_parameter(cfg["url"], "list") == get_url_parameter(config["url"], "list"):
                            print(f"Playlist '{p['playlist_name']}' already downloaded.")
                            if get_bool_option_response("Update playlist?", True):
                                config = cfg
                                current_playlist_name = p["playlist_name"]
                                generate_playlist(config, config_file_name, True, False, False, False, current_playlist_name)
                            break
                    else:
                        config = validate_config(config)
                        config["reverse_playlist"] = get_bool_option_response("Reverse playlist?", False)
                        config["use_title"] = get_bool_option_response("Use title instead of track name?", True)
                        config["use_uploader"] = get_bool_option_response("Use uploader instead of artist?", True)
                        config["use_playlist_name"] = get_bool_option_response("Use playlist name for album?", True)
                        generate_playlist(config, config_file_name, False, False, False, False)

                elif action == "update":
                    idx = get_index_option_response("Select playlist to update", len(playlists_data), True)
                    if idx == "all":
                        for p in playlists_data:
                            with Path(p["config_file"]).open() as f:
                                config = validate_config(json.load(f))
                            generate_playlist(config, config_file_name, True, False, False, False, p["playlist_name"])
                    else:
                        p = playlists_data[idx]
                        with Path(p["config_file"]).open() as f:
                            config = validate_config(json.load(f))
                        generate_playlist(config, config_file_name, True, False, False, False, p["playlist_name"])

                elif action == "song":
                    idx = get_index_option_response("Select playlist", len(playlists_data))
                    p = playlists_data[idx]
                    with Path(p["config_file"]).open() as f:
                        config = validate_config(json.load(f))
                    track = get_numeric_option_response("Enter song track number")
                    generate_playlist(config, config_file_name, True, False, False, False, p["playlist_name"], track)

                elif action == "modify":
                    idx = get_index_option_response("Select playlist", len(playlists_data))
                    p = playlists_data[idx]
                    with Path(p["config_file"]).open() as f:
                        config = validate_config(json.load(f))
                    print(f"Playlist: {p['playlist_name']} - {config['url']}")
                    if get_bool_option_response("Change settings?", False):
                        config["reverse_playlist"] = get_bool_option_response("Reverse playlist?", config["reverse_playlist"])
                        config["use_title"] = get_bool_option_response("Use title instead of track name?", config["use_title"])
                        config["use_uploader"] = get_bool_option_response("Use uploader instead of artist?", config["use_uploader"])
                        config["use_playlist_name"] = get_bool_option_response("Use playlist name for album?", config["use_playlist_name"])
                    force = get_bool_option_response("Force update all metadata?", False)
                    generate_playlist(config, config_file_name, True, force, False, False, p["playlist_name"])

                elif action == "generate":
                    config["url"] = input("Enter playlist URL for config: ")
                    generate_playlist(config, config_file_name, False, False, False, False)

                elif action == "change":
                    os.chdir(input("Enter new directory path: "))

                elif action == "exit":
                    break

            if single_playlist:
                generate_playlist(config, config_file_name, True, False, False, True, current_playlist_name)
            input("Press Enter to continue...")
        except KeyboardInterrupt:
            logger.info("Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            input("Press Enter to retry...")