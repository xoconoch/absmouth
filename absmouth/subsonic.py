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

    def get_artists(self):
        sub_resp = self.request("getArtists.view")
        artists_data = sub_resp.get("artists", {})
        indices = artists_data.get("index", [])
        if not isinstance(indices, list):
            indices = [indices]
        
        all_artists = []
        for index in indices:
            artists = index.get("artist", [])
            if not isinstance(artists, list):
                artists = [artists]
            all_artists.extend(artists)
        return all_artists

    def get_artist(self, artist_id):
        sub_resp = self.request("getArtist.view", {"id": artist_id})
        artist = sub_resp.get("artist", {})
        if not artist:
            return []
        albums = artist.get("album", [])
        if albums is None:
            return []
        if not isinstance(albums, list):
            albums = [albums]
        return albums

    def get_album(self, album_id):
        sub_resp = self.request("getAlbum.view", {"id": album_id})
        album = sub_resp.get("album", {})
        if not album:
            return []
        songs = album.get("song", [])
        if songs is None:
            return []
        if not isinstance(songs, list):
            songs = [songs]
        return songs


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
