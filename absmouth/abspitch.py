import requests

class AbsolutePitchClient:
    """Client to insert track metadata and embeddings to the Absolute Pitch backend."""
    def __init__(self, config):
        self.api_url = config.API_URL
        self.request_timeout = getattr(config, "API_REQUEST_TIMEOUT", 30)

    def insert_track(self, subsonic_id, title, album_name, artist_id, artist_name, embedding):
        """
        Sends the 1024-d embedding and track metadata to Absolute Pitch.
        Returns the HTTP response.
        """
        payload = {
            "subsonic_id": subsonic_id,
            "title": title,
            "album_name": album_name,
            "artist_id": artist_id,
            "artist_name": artist_name,
            "embedding": embedding
        }
        
        response = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
        return response

    def check_track(self, subsonic_id):
        """
        Queries the Absolute Pitch backend to check if a track has already been processed.
        Returns the HTTP response.
        """
        url = f"{self.api_url.rstrip('/')}/{subsonic_id}"
        return requests.get(url, timeout=self.request_timeout)

    def get_processed_track_ids(self, page_size=500):
        """
        Fetches the IDs of every track already stored on the backend by paging
        through the collection endpoint. Returns a set of subsonic_id strings.

        Raises on request or payload errors so the caller can fall back to
        per-track checks.
        """
        url = self.api_url.rstrip("/")
        processed = set()
        offset = 0

        while True:
            response = requests.get(
                url,
                params={"limit": page_size, "offset": offset},
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            page = response.json()
            if not isinstance(page, list):
                raise ValueError(f"Expected a list of tracks, got {type(page).__name__}")

            if not page:
                break

            new_ids = {
                str(track["subsonic_id"])
                for track in page
                if isinstance(track, dict) and track.get("subsonic_id") is not None
            }
            new_ids -= processed
            if not new_ids:
                # Server ignored the offset (or repeated a page); stop instead of looping.
                break

            processed |= new_ids
            offset += len(page)

        return processed

