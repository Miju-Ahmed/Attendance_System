#!/usr/bin/env python3
"""
Debug script to test line crossing detection logic
"""

def get_side(x1, y1, x2, y2, px, py):
    """Test the line side detection"""
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    
    print(f"  Cross product: {cross}")
    
    if cross > 0.1:
        return 1
    elif cross < -0.1:
        return -1
    else:
        return 0

def check_crossing(prev_side, curr_side):
    """Check if a crossing occurred"""
    if prev_side == 0 or curr_side == 0:
        return None
    
    if prev_side < 0 and curr_side > 0:
        return "ENTRY"
    elif prev_side > 0 and curr_side < 0:
        return "EXIT"
    
    return None

# Test scenarios
print("Testing Line Crossing Detection")
print("=" * 60)

# Example line: from (100, 300) to (500, 300) - horizontal line
line_x1, line_y1 = 100, 300
line_x2, line_y2 = 500, 300

print(f"\nLine: ({line_x1}, {line_y1}) to ({line_x2}, {line_y2})")
print("-" * 60)

# Test points
test_points = [
    (300, 250, "Above line (OUTSIDE)"),
    (300, 300, "On line"),
    (300, 350, "Below line (INSIDE)"),
]

print("\nPoint positions:")
for px, py, desc in test_points:
    side = get_side(line_x1, line_y1, line_x2, line_y2, px, py)
    side_name = "OUTSIDE" if side < 0 else ("INSIDE" if side > 0 else "ON LINE")
    print(f"  ({px}, {py}) - {desc}: side={side} ({side_name})")

print("\n" + "=" * 60)
print("Crossing Scenarios:")
print("-" * 60)

scenarios = [
    (-1, 1, "OUTSIDE → INSIDE (should be ENTRY)"),
    (1, -1, "INSIDE → OUTSIDE (should be EXIT)"),
    (-1, -1, "OUTSIDE → OUTSIDE (no crossing)"),
    (1, 1, "INSIDE → INSIDE (no crossing)"),
    (0, 1, "ON LINE → INSIDE (ignored)"),
    (-1, 0, "OUTSIDE → ON LINE (ignored)"),
]

for prev, curr, desc in scenarios:
    event = check_crossing(prev, curr)
    result = event if event else "None"
    print(f"  {desc}: {result}")

print("\n" + "=" * 60)
