from library.monitor_client import MonitorClient

client = MonitorClient()

print("STATS:")
print(client.get_stats())

print("\nALL MESSAGES:")
msgs = client.get_messages()
print(len(msgs))

print("\nSERVICES:")
print(client.get_services())

print("\nHISTORY for service 'camera1':")
print(client.get_history("example_service2"))

print("Status of service example_service2")
print(client.get_service("camera1"))