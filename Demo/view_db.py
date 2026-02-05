#!/usr/bin/env python3
"""
Quick script to verify attendance database records
"""
import sqlite3
from datetime import datetime

DB_PATH = "attendance.db"

def view_attendance():
    """View all attendance records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM attendance")
        total = cursor.fetchone()[0]
        
        print("=" * 80)
        print(f"ATTENDANCE DATABASE - Total Records: {total}")
        print("=" * 80)
        
        if total == 0:
            print("\n⚠️  No records found in database yet.")
            print("   Run the attendance system and cross the line to generate records.\n")
            return
        
        # Get all records ordered by timestamp
        cursor.execute("""
            SELECT id, person_name, event_type, timestamp, confidence, track_id
            FROM attendance
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        records = cursor.fetchall()
        
        print("\nRecent Records (newest first):")
        print("-" * 80)
        print(f"{'ID':<5} {'Name':<15} {'Event':<8} {'Timestamp':<20} {'Conf':<6} {'Track':<6}")
        print("-" * 80)
        
        for record in records:
            id_, name, event, timestamp, conf, track_id = record
            conf_str = f"{conf:.2f}" if conf else "N/A"
            print(f"{id_:<5} {name:<15} {event:<8} {timestamp:<20} {conf_str:<6} {track_id:<6}")
        
        print("-" * 80)
        
        # Statistics
        cursor.execute("SELECT event_type, COUNT(*) FROM attendance GROUP BY event_type")
        stats = cursor.fetchall()
        
        print("\nStatistics:")
        for event_type, count in stats:
            print(f"  {event_type}: {count}")
        
        print("\n")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    view_attendance()
