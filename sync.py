"""
sync.py — Cloud version of LLM Database Sync
Runs on Railway as a cron job. Uses PostgreSQL for sync state persistence.
"""

import os, json, io, time, traceback, urllib.request
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

# ─── CONFIGURATION (from environment variables) ───────────────────────────────

SHARED_DRIVE_NAMES = [
    "LLM Database",
    "Rossmonster Builds",
    "marketing",
    "Rossmonster CAD",
]

PINECONE_INDEX    = os.environ.get("PINECONE_INDEX", "rossmonster-llm-db")
PINECONE_REGION   = os.environ.get("PINECONE_REGION", "us-east-1")
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

def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX}'...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX)

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

# ─── DRIVE HELPERS ────────────────────────────────────────────────────────────

def find_shared_drive_id(service, name):
    result = service.drives().list(pageSize=50).execute()
    for drive in result.get("drives", []):
        if drive["name"] == name:
            return drive["id"]
    raise ValueError(f"Shared Drive '{name}' not found. Make sure the service account has been added as a Viewer.")

def list_all_files(service, drive_id):
    files = []
    page_token = None
    while True:
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
            return "\n".join(page.extract_text() or "" for page in reader.pages)
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

def embed_texts(texts):
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = openai_client.embeddings.create(input=batch, model=EMBED_MODEL)
        all_vectors.extend([item.embedding for item in response.data])
    return all_vectors

# ─── SLACK NOTIFICATION ───────────────────────────────────────────────────────

def send_slack(success, upserted, skipped, error=None):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        if success:
            text = (
                f"✅ *LLM Database Sync Complete*\n"
                f"• Vectors upserted: {upserted:,}\n"
                f"• Files skipped: {skipped:,} (already up to date)\n"
                f"• Drives: {', '.join(SHARED_DRIVE_NAMES)}\n"
                f"• Completed: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
            )
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

    upserted_total = 0
    skipped_total  = 0

    for drive_name in SHARED_DRIVE_NAMES:
        print(f"\n🔍 Finding Shared Drive: '{drive_name}'...")
        try:
            drive_id = find_shared_drive_id(service, drive_name)
            print(f"   Found: {drive_id}")
        except ValueError as e:
            print(f"   ⚠️  Skipping: {e}")
            continue

        print("📂 Listing all files...")
        files = list_all_files(service, drive_id)
        print(f"   Found {len(files)} files")

        for file in tqdm(files, desc=f"Processing '{drive_name}'"):
            if file.get("mimeType") == "application/vnd.google-apps.shortcut":
                file = resolve_shortcut(service, file)
                if not file:
                    continue

            fid      = file["id"]
            fname    = file["name"]
            modified = file["modifiedTime"]
            mime     = file["mimeType"]

            if mime == "application/vnd.google-apps.folder":
                continue

            if sync_state.get(fid) == modified:
                skipped_total += 1
                continue

            text = extract_text(service, file)
            if not text or not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            try:
                vectors = embed_texts(chunks)
            except Exception as e:
                print(f"  ⚠️  Embedding failed for '{fname}': {e}")
                continue

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

            # Save progress to DB after each file (not just at the end)
            save_file_state(fid, modified)
            sync_state[fid] = modified
            upserted_total += len(records)

    print(f"\n✅ Sync complete!")
    print(f"   Vectors upserted : {upserted_total}")
    print(f"   Files skipped    : {skipped_total}")
    send_slack(success=True, upserted=upserted_total, skipped=skipped_total)

if __name__ == "__main__":
    try:
        sync()
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ Sync failed: {e}")
        send_slack(success=False, upserted=0, skipped=0, error=str(e))
        raise
