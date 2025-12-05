import asyncio
from architecture.output_layer_monitor import monitor

async def run():
    await monitor.monitor.connect()
    await monitor._receiver_loop()

if __name__ == "__main__":
    asyncio.run(run())