"""Workflow: 10 Frames Face Extractor → NATS → Gaze → Kafka"""
import asyncio


async def run_workflow():
    from gaze_nats_producer import run_producer
    from simple_consumer import run_consumer
    
    print("🚀 Pipeline: Face Extractor → NATS → Gaze → Kafka (10 frames)")
    
    await asyncio.gather(
        run_producer(10),
        run_consumer(10)
    )
    
    print("\n✅ Pipeline Complete!")


if __name__ == "__main__":
    asyncio.run(run_workflow())
