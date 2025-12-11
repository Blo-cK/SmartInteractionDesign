import asyncio
from output_layer_monitor import monitor

async def run():
    await monitor.monitor.connect()
    print("Connection done")
    await monitor._receiver_loop()

if __name__ == "__main__":
    asyncio.run(run())