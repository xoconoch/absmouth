import hashlib
import random
import string
import requests

class SubsonicClient:
    """Client to query Subsonic server, crawl tracks, and stream audio downloads."""
    def __init__(self, config):
        self.url = config.SUBSONIC_URL.rstrip("/")
        self.username = config.SUBSONIC_USER
        self.password = config.SUBSONIC_PASS
        self.version = config.SUBSONIC_API_VERSION
        self.client_name = config.SUBSONIC_CLIENT_NAME

    def _get_auth_params(self):
        salt = "".join(random.choices(string.ascii_letters + string.digits, k=12))
        token = hashlib.md5((self.password + salt).encode('utf-8')).hexdigest()
        return {
            "u": self.username,
            "t": token,
            "s": salt,
            "v": self.version,
            "c": self.client_name,
            "f": "json"
        }

    def request(self, endpoint, additional_params=None):
        url = f"{self.url}/rest/{endpoint}"
        params = self._get_auth_params()
        if additional_params:
            params.update(additional_params)
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        sub_resp = data.get("subsonic-response", {})
        if sub_resp.get("status") == "failed":
            error_info = sub_resp.get("error", {})
            raise Exception(f"Subsonic error {error_info.get('code')}: {error_info.get('message')}")
        
        return sub_resp

    def get_music_folders(self):
        sub_resp = self.request("getMusicFolders.view")
        folders = sub_resp.get("musicFolders", {}).get("musicFolder", [])
        if not isinstance(folders, list):
            folders = [folders]
        return folders

    def get_music_directory(self, dir_id):
        sub_resp = self.request("getMusicDirectory.view", {"id": dir_id})
        directory = sub_resp.get("directory", {})
        children = directory.get("child", [])
        if not isinstance(children, list):
            children = [children]
        return children

    def crawl_all_tracks(self):
        """Recursively walks through Subsonic folders to build a list of all audio files."""
        print("Crawling Subsonic directory structure...")
        try:
            folders = self.get_music_folders()
        except Exception as e:
            print(f"Error fetching music folders from Subsonic: {e}")
            return []

        all_tracks = []
        visited_dirs = set()

        def crawl(dir_id):
            if dir_id in visited_dirs:
                return
            visited_dirs.add(dir_id)
            try:
                children = self.get_music_directory(dir_id)
                for child in children:
                    if child.get("isDir", False):
                        crawl(child.get("id"))
                    else:
                        all_tracks.append(child)
            except Exception as e:
                print(f"Warning: Failed to crawl directory {dir_id}: {e}")

        for folder in folders:
            crawl(folder.get("id"))

        print(f"Crawling complete. Found {len(all_tracks)} total songs/files.")
        return all_tracks

    def download_track(self, track_id, dest_path):
        """Streams track download directly to disk to preserve memory usage."""
        params = self._get_auth_params()
        params["id"] = track_id
        url = f"{self.url}/rest/stream.view"
        
        with requests.get(url, params=params, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=16384):
                    f.write(chunk)
