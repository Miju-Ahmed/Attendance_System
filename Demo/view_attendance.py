"""
View Attendance Records
========================
Simple script to view attendance records from the database.
"""

import sqlite3
import argparse
from datetime import datetime, timedelta
from tabulate import tabulate


def view_all_records(db_path: str, limit: int = 50):
    """View all attendance records."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, person_name, event_type, timestamp, confidence, track_id
        FROM attendance
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print("No attendance records found.")
        return
    
    headers = ["ID", "Name", "Event", "Timestamp", "Confidence", "Track ID"]
    print(f"\n📊 Recent Attendance Records (Last {limit}):\n")
    print(tabulate(records, headers=headers, tablefmt="grid"))


def view_summary(db_path: str):
    """View attendance summary."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Total counts
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE event_type = 'ENTRY'")
    total_entries = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE event_type = 'EXIT'")
    total_exits = cursor.fetchone()[0]
    
    # Per person summary
    cursor.execute("""
        SELECT 
            person_name,
            SUM(CASE WHEN event_type = 'ENTRY' THEN 1 ELSE 0 END) as entries,
            SUM(CASE WHEN event_type = 'EXIT' THEN 1 ELSE 0 END) as exits
        FROM attendance
        GROUP BY person_name
        ORDER BY entries DESC
    """)
    
    person_summary = cursor.fetchall()
    
    conn.close()
    
    print("\n" + "="*60)
    print("ATTENDANCE SUMMARY")
    print("="*60)
    print(f"Total Entries: {total_entries}")
    print(f"Total Exits: {total_exits}")
    print("="*60)
    
    if person_summary:
        print("\n📈 Per Person Summary:\n")
        headers = ["Name", "Entries", "Exits"]
        print(tabulate(person_summary, headers=headers, tablefmt="grid"))
    else:
        print("\nNo person-specific data available.")


def view_today(db_path: str):
    """View today's attendance."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    cursor.execute("""
        SELECT id, person_name, event_type, timestamp, confidence, track_id
        FROM attendance
        WHERE DATE(timestamp) = ?
        ORDER BY timestamp DESC
    """, (today,))
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print(f"\nNo attendance records found for today ({today}).")
        return
    
    headers = ["ID", "Name", "Event", "Timestamp", "Confidence", "Track ID"]
    print(f"\n📅 Today's Attendance ({today}):\n")
    print(tabulate(records, headers=headers, tablefmt="grid"))


def view_person(db_path: str, name: str):
    """View attendance for a specific person."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, person_name, event_type, timestamp, confidence, track_id
        FROM attendance
        WHERE person_name LIKE ?
        ORDER BY timestamp DESC
    """, (f"%{name}%",))
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print(f"\nNo attendance records found for: {name}")
        return
    
    headers = ["ID", "Name", "Event", "Timestamp", "Confidence", "Track ID"]
    print(f"\n👤 Attendance for '{name}':\n")
    print(tabulate(records, headers=headers, tablefmt="grid"))


def clear_database(db_path: str, confirm: bool = False):
    """Clear all attendance records."""
    if not confirm:
        print("\n⚠️  Warning: This will delete all attendance records!")
        response = input("Type 'DELETE ALL' to confirm: ")
        if response != "DELETE ALL":
            print("Operation cancelled.")
            return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM attendance")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"\n✅ Deleted {deleted} attendance records.")


def export_to_csv(db_path: str, output_file: str):
    """Export attendance to CSV."""
    import csv
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, person_name, event_type, timestamp, confidence, track_id
        FROM attendance
        ORDER BY timestamp
    """)
    
    records = cursor.fetchall()
    conn.close()
    
    if not records:
        print("No records to export.")
        return
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Event", "Timestamp", "Confidence", "Track ID"])
        writer.writerows(records)
    
    print(f"\n✅ Exported {len(records)} records to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="View Attendance Records")
    parser.add_argument(
        "--db",
        type=str,
        default="attendance.db",
        help="Path to attendance database"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="View all records"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="View summary"
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="View today's records"
    )
    parser.add_argument(
        "--person",
        type=str,
        help="View records for specific person"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of records to show"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export to CSV file"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all attendance records"
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    import os
    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        return
    
    # Execute command
    if args.clear:
        clear_database(args.db)
    elif args.export:
        export_to_csv(args.db, args.export)
    elif args.person:
        view_person(args.db, args.person)
    elif args.today:
        view_today(args.db)
    elif args.summary:
        view_summary(args.db)
    elif args.all:
        view_all_records(args.db, args.limit)
    else:
        # Default: show summary
        view_summary(args.db)
        print("\n")
        view_all_records(args.db, limit=10)


if __name__ == "__main__":
    main()
