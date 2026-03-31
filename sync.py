"""
sync.py — Cloud version of LLM Database Sync
Runs on Railway as a cron job. Uses PostgreSQL for sync state persistence.

Uses the Google Drive Changes API for incremental syncs — only processes
files that have actually changed since the last run instead of listing
all 60,000+ files every time.
"""

import os, json, io, time, traceback, urllib.request, re
from datetime import datetime

import psycopg2
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import docx as python_docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ─── CONFIGURATION (from environment variables) ───────────────────────────────

SHARED_DRIVE_NAMES = [
    "LLM Database",
    "Rossmonster Builds",
    "marketing",
    "Rossmonster CAD",
    "Sales",
]

PINECONE_INDEX         = os.environ.get("PINECONE_INDEX", "rossmonster-llm-db")
PINECONE_INDEX_ARCHIVE = os.environ.get("PINECONE_INDEX_ARCHIVE", PINECONE_INDEX + "-archive")
PINECONE_REGION        = os.environ.get("PINECONE_REGION", "us-east-1")
EMBED_MODEL       = "text-embedding-3-small"
EMBED_DIMENSIONS  = 1536
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# ─── CLIENTS ──────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

_pc = None

def get_pc():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pc

def _ensure_index(name):
    pc = get_pc()
    existing = [i.name for i in pc.list_indexes()]
    if name not in existing:
        print(f"Creating Pinecone index '{name}'...")
        pc.create_index(
            name=name,
            dimension=EMBED_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)

def get_pinecone_index():
    return _ensure_index(PINECONE_INDEX)

def get_pinecone_archive_index():
    return _ensure_index(PINECONE_INDEX_ARCHIVE)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── POSTGRES SYNC STATE ──────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(os.environ["DATABASE_URL"])

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sync_state (
                    file_id TEXT PRIMARY KEY,
                    modified_time TEXT NOT NULL
                )
            """)
            # Stores the Drive Changes API page token per drive
            cur.execute("""
                CREATE TABLE IF NOT EXISTS drive_tokens (
                    drive_id TEXT PRIMARY KEY,
                    page_token TEXT NOT NULL
                )
            """)
        conn.commit()

def load_sync_state():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT file_id, modified_time FROM sync_state")
            return {row[0]: row[1] for row in cur.fetchall()}

def save_file_state(file_id, modified_time):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sync_state (file_id, modified_time)
                VALUES (%s, %s)
                ON CONFLICT (file_id) DO UPDATE SET modified_time = EXCLUDED.modified_time
            """, (file_id, modified_time))
        conn.commit()

def remove_file_state(file_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sync_state WHERE file_id = %s", (file_id,))
        conn.commit()

def load_drive_token(drive_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT page_token FROM drive_tokens WHERE drive_id = %s", (drive_id,))
            row = cur.fetchone()
            return row[0] if row else None

def save_drive_token(drive_id, page_token):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO drive_tokens (drive_id, page_token)
                VALUES (%s, %s)
                ON CONFLICT (drive_id) DO UPDATE SET page_token = EXCLUDED.page_token
            """, (drive_id, page_token))
        conn.commit()

# ─── DRIVE HELPERS ────────────────────────────────────────────────────────────

def find_shared_drive_id(service, name):
    result = service.drives().list(pageSize=50).execute()
    for drive in result.get("drives", []):
        if drive["name"] == name:
            return drive["id"]
    raise ValueError(f"Shared Drive '{name}' not found. Make sure the service account has been added as a Viewer.")

def get_start_page_token(service, drive_id):
    """Get the current Changes API page token for a drive (marks 'start tracking from now')."""
    resp = service.changes().getStartPageToken(
        driveId=drive_id,
        supportsAllDrives=True,
    ).execute()
    return resp.get("startPageToken")

def get_drive_changes(service, drive_id, page_token):
    """
    Fetch all changes since page_token using the Drive Changes API.
    Returns (list_of_changes, new_start_token).
    Each change has: fileId, removed (bool), file (dict or None if removed).
    """
    changes = []
    current_token = page_token
    while True:
        for attempt in range(3):
            try:
                resp = service.changes().list(
                    pageToken=current_token,
                    driveId=drive_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    spaces="drive",
                    fields="nextPageToken, newStartPageToken, changes(fileId, removed, file(id, name, mimeType, modifiedTime, shortcutDetails))",
                    pageSize=1000,
                ).execute()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  ⚠️  Changes API error (attempt {attempt+1}/3): {e} — retrying in 5s...")
                time.sleep(5)

        changes.extend(resp.get("changes", []))

        if "nextPageToken" in resp:
            current_token = resp["nextPageToken"]
        else:
            new_token = resp.get("newStartPageToken")
            return changes, new_token

def list_all_files(service, drive_id):
    """Full file listing — only used on first run for a drive."""
    files = []
    page_token = None
    while True:
        for attempt in range(3):
            try:
                resp = service.files().list(
                    q="trashed = false",
                    corpora="drive",
                    driveId=drive_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, parents, shortcutDetails)",
                    pageSize=200,
                    pageToken=page_token,
                ).execute()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  ⚠️  Google API error (attempt {attempt+1}/3): {e} — retrying in 5s...")
                time.sleep(5)
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files

def resolve_shortcut(service, file):
    if file.get("mimeType") != "application/vnd.google-apps.shortcut":
        return file
    try:
        target_id = file.get("shortcutDetails", {}).get("targetId")
        if not target_id:
            full = service.files().get(fileId=file["id"], fields="shortcutDetails", supportsAllDrives=True).execute()
            target_id = full.get("shortcutDetails", {}).get("targetId")
        if not target_id:
            return None
        return service.files().get(fileId=target_id, fields="id, name, mimeType, modifiedTime", supportsAllDrives=True).execute()
    except Exception as e:
        print(f"  ⚠️  Could not resolve shortcut '{file.get('name')}': {e}")
        return None

def export_google_doc(service, file_id):
    request = service.files().export_media(fileId=file_id, mimeType="text/plain")
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue().decode("utf-8", errors="ignore")

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf

def extract_text(service, file):
    mime = file["mimeType"]
    fid  = file["id"]
    try:
        if mime == "application/vnd.google-apps.document":
            return export_google_doc(service, fid)
        elif mime == "application/vnd.google-apps.spreadsheet":
            request = service.files().export_media(fileId=fid, mimeType="text/csv")
            buf = io.BytesIO()
            MediaIoBaseDownload(buf, request).next_chunk()
            return buf.getvalue().decode("utf-8", errors="ignore")
        elif mime == "application/vnd.google-apps.presentation":
            request = service.files().export_media(fileId=fid, mimeType="text/plain")
            buf = io.BytesIO()
            MediaIoBaseDownload(buf, request).next_chunk()
            return buf.getvalue().decode("utf-8", errors="ignore")
        elif mime == "application/pdf" and HAS_PYPDF:
            buf = download_file(service, fid)
            reader = pypdf.PdfReader(buf)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                print(f"  ℹ️  PDF '{file['name']}' returned no extractable text — likely a scanned image, skipping")
            return text
        elif mime in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ) and HAS_OPENPYXL:
            buf = download_file(service, fid)
            wb = openpyxl.load_workbook(buf, read_only=True, data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"[Sheet: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        parts.append(row_text)
            return "\n".join(parts) if parts else None
        elif mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) and HAS_DOCX:
            buf = download_file(service, fid)
            doc = python_docx.Document(buf)
            return "\n".join(p.text for p in doc.paragraphs)
        elif mime.startswith("text/"):
            buf = download_file(service, fid)
            return buf.read().decode("utf-8", errors="ignore")
        else:
            return None
    except Exception as e:
        print(f"  ⚠️  Could not extract text from '{file['name']}': {e}")
        return None

# ─── CHUNKING & EMBEDDING ─────────────────────────────────────────────────────

def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def sanitize_text(text):
    """Remove control characters that break JSON serialization."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def embed_texts(texts):
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = [sanitize_text(t) for t in texts[i:i + 100]]
        response = openai_client.embeddings.create(input=batch, model=EMBED_MODEL)
        all_vectors.extend([item.embedding for item in response.data])
    return all_vectors

def process_file(service, index, file, drive_name, sync_state):
    """
    Embed and upsert a single file into Pinecone.
    Returns number of vectors upserted (0 if skipped or failed).
    """
    if file.get("mimeType") == "application/vnd.google-apps.shortcut":
        file = resolve_shortcut(service, file)
        if not file:
            return 0

    fid      = file["id"]
    fname    = file["name"]
    modified = file["modifiedTime"]
    mime     = file["mimeType"]

    if mime == "application/vnd.google-apps.folder":
        return 0

    if sync_state.get(fid) == modified:
        return -1  # unchanged

    text = extract_text(service, file)
    if not text or not text.strip():
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    try:
        vectors = embed_texts(chunks)
    except Exception as e:
        print(f"  ⚠️  Embedding failed for '{fname}': {e}")
        return 0

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": f"{fid}_chunk_{i}",
            "values": vector,
            "metadata": {
                "file_id":       fid,
                "file_name":     fname,
                "mime_type":     mime,
                "modified_time": modified,
                "chunk_index":   i,
                "total_chunks":  len(chunks),
                "text":          chunk,
                "drive_name":    drive_name,
            },
        })

    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i:i+100])

    save_file_state(fid, modified)
    sync_state[fid] = modified
    return len(records)

def archive_file_vectors(index, archive_index, file_id):
    """
    Move all Pinecone vectors for a deleted Drive file to the archive index
    instead of hard-deleting them. Uses chunk ID prefix listing.
    Returns the number of vectors archived.
    """
    removed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    chunk_ids = []
    try:
        for id_batch in index.list(prefix=f"{file_id}_chunk_"):
            chunk_ids.extend(id_batch)
    except Exception as e:
        print(f"  ⚠️  Could not list vectors for {file_id}: {e}")
        # Fall back to metadata-filter delete so nothing is left dangling
        try:
            index.delete(filter={"file_id": file_id})
        except Exception:
            pass
        return 0

    if not chunk_ids:
        return 0

    archived_count = 0
    for i in range(0, len(chunk_ids), 100):
        batch = chunk_ids[i:i + 100]
        try:
            result = index.fetch(ids=batch)
        except Exception as e:
            print(f"  ⚠️  Fetch failed for batch of {file_id}: {e}")
            continue

        if not result.vectors:
            continue

        archive_records = []
        for vid, vec in result.vectors.items():
            meta = dict(vec.metadata)
            meta["removed_at"] = removed_at
            archive_records.append({
                "id": vid,
                "values": list(vec.values),
                "metadata": meta,
            })

        try:
            archive_index.upsert(vectors=archive_records)
            index.delete(ids=batch)
            archived_count += len(archive_records)
        except Exception as e:
            print(f"  ⚠️  Archive upsert/delete failed: {e}")

    return archived_count

# ─── SLACK NOTIFICATION ───────────────────────────────────────────────────────

def send_slack(success, upserted, skipped, deleted=0, full_sync_drives=None, error=None):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        if success:
            mode = "Full sync" if full_sync_drives else "Incremental sync"
            text = (
                f"✅ *LLM Database Sync Complete* ({mode})\n"
                f"• Vectors upserted: {upserted:,}\n"
                f"• Files unchanged: {skipped:,}\n"
                f"• Files archived (not deleted): {deleted:,}\n"
            )
            if full_sync_drives:
                text += f"• First-time full scan: {', '.join(full_sync_drives)}\n"
            text += f"• Completed: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
        else:
            text = (
                f"❌ *LLM Database Sync Failed*\n"
                f"• Error: {error}\n"
                f"• Possible causes: expired API key, lost Drive access, network issue, Pinecone unavailable"
            )
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(SLACK_WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req)
        print(f"  💬 Slack notification sent")
    except Exception as e:
        print(f"  ⚠️  Failed to send Slack notification: {e}")

# ─── MAIN SYNC ────────────────────────────────────────────────────────────────

def sync():
    init_db()

    print("🔌 Connecting to Google Drive...")
    service = get_drive_service()

    print("🗄️  Connecting to Pinecone...")
    index = get_pinecone_index()

    print("📋 Loading sync state from database...")
    sync_state = load_sync_state()
    print(f"   {len(sync_state):,} files already synced")

    print("📦 Connecting to Pinecone archive index...")
    archive_index = get_pinecone_archive_index()

    upserted_total  = 0
    skipped_total   = 0
    archived_total  = 0
    full_sync_drives = []

    for drive_name in SHARED_DRIVE_NAMES:
        print(f"\n🔍 Finding Shared Drive: '{drive_name}'...")
        try:
            drive_id = find_shared_drive_id(service, drive_name)
            print(f"   Found: {drive_id}")
        except ValueError as e:
            print(f"   ⚠️  Skipping: {e}")
            continue

        stored_token = load_drive_token(drive_id)

        if stored_token is None:
            # ── FIRST RUN: full file scan ──────────────────────────────────
            print(f"   No Changes token found — doing full scan (first time only)...")
            full_sync_drives.append(drive_name)

            # Grab the start token BEFORE listing so we don't miss any changes
            # that happen during this initial scan
            start_token = get_start_page_token(service, drive_id)

            print("📂 Listing all files...")
            files = list_all_files(service, drive_id)
            print(f"   Found {len(files):,} files")

            for file in tqdm(files, desc=f"Full scan '{drive_name}'"):
                result = process_file(service, index, file, drive_name, sync_state)
                if result > 0:
                    upserted_total += result
                elif result == -1:
                    skipped_total += 1

            save_drive_token(drive_id, start_token)
            print(f"   ✅ Full scan complete — Changes token saved")

        else:
            # ── INCREMENTAL: only changed files ───────────────────────────
            print(f"   Changes token found — fetching incremental changes...")
            changes, new_token = get_drive_changes(service, drive_id, stored_token)

            # De-duplicate: if the same file changed multiple times, keep last
            # Skip drive-level changes that have no fileId
            seen = {}
            for change in changes:
                if "fileId" not in change:
                    continue
                seen[change["fileId"]] = change
            changes = list(seen.values())

            added_or_modified = [c for c in changes if not c.get("removed") and c.get("file")]
            removed = [c for c in changes if c.get("removed")]

            print(f"   {len(added_or_modified)} modified/new files, {len(removed)} deleted")

            for change in tqdm(added_or_modified, desc=f"Incremental '{drive_name}'"):
                file = change["file"]
                result = process_file(service, index, file, drive_name, sync_state)
                if result > 0:
                    upserted_total += result
                elif result == -1:
                    skipped_total += 1

            for change in removed:
                fid = change["fileId"]
                print(f"   📦 Archiving vectors for removed file: {fid}")
                count = archive_file_vectors(index, archive_index, fid)
                print(f"      Archived {count} vectors")
                remove_file_state(fid)
                if fid in sync_state:
                    del sync_state[fid]
                archived_total += 1

            save_drive_token(drive_id, new_token)

    print(f"\n✅ Sync complete!")
    print(f"   Vectors upserted : {upserted_total:,}")
    print(f"   Files unchanged  : {skipped_total:,}")
    print(f"   Files archived   : {archived_total:,} (moved to '{PINECONE_INDEX_ARCHIVE}')")
    send_slack(
        success=True,
        upserted=upserted_total,
        skipped=skipped_total,
        deleted=archived_total,
        full_sync_drives=full_sync_drives if full_sync_drives else None,
    )

if __name__ == "__main__":
    try:
        sync()
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ Sync failed: {e}")
        send_slack(success=False, upserted=0, skipped=0, error=str(e))
        raise
