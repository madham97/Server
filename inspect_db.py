import sqlite3
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db', default='objects.db')
args = parser.parse_args()

conn = sqlite3.connect(args.db)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print('Objects:')
for r in c.execute('SELECT * FROM objects ORDER BY id'):
    print(dict(r))

print('\nActive tracks:')
for r in c.execute('SELECT * FROM active_tracks'):
    d = dict(r)
    d['last_bbox'] = json.loads(d['last_bbox']) if d['last_bbox'] else None
    print(d)

print('\nRecent sightings (limit 50):')
for r in c.execute('SELECT * FROM sightings ORDER BY id DESC LIMIT 50'):
    d = dict(r)
    d['bbox'] = json.loads(d['bbox']) if r['bbox'] else None
    print(d)

conn.close()