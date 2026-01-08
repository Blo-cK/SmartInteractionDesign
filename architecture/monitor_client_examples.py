from library.monitor_client import MonitorClient, MonitorAPIError

client = MonitorClient()

print("=== STATS ===")
try:
    stats = client.get_stats()
    print(stats)
except MonitorAPIError as e:
    print("Error fetching stats:", e)


print("\n=== ALL MESSAGES ===")
try:
    msgs = client.get_messages()
    print(f"Total messages: {len(msgs)}")
    if msgs:
        print("Example message:", msgs[0])
except MonitorAPIError as e:
    print("Error fetching messages:", e)


print("\n=== SERVICES ===")
try:
    services = client.get_services()
    for s in services:
        print(s)
except MonitorAPIError as e:
    print("Error fetching services:", e)


print("\n=== HISTORY for output service 'example_service2' ===")
try:
    hist = client.get_history("example_service2")
    if hist:
        print(f"History entries: {len(hist)}")
        print("First entry:", hist[0])
    else:
        print("No history found.")
except MonitorAPIError as e:
    print("Error fetching history:", e)


print("\n=== Status of service 'camera1' ===")
try:
    status = client.get_online_status("camera1")
    if status is None:
        print("Service 'camera1' not found")
    else:
        print("Entity: ", status)
        print("Last seen:", status.last_seen)
        print("Online:", status.online)
except MonitorAPIError as e:
    print("Error fetching service status:", e)
