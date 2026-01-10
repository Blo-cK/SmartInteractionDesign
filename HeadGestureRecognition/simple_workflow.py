"""Workflow: 10 Frames Face Extractor → NATS → HeadGesture → Kafka"""
import asyncio


async def run_workflow():
    from headgesture_nats_producer import run_producer
    from simple_consumer import run_consumer
    
    print("🚀 Pipeline: Face Extractor → NATS → HeadGesture → Kafka (10 frames)")
    
    await asyncio.gather(
        run_producer(-1),
        run_consumer()
    )
    
    print("\n✅ Pipeline Complete!")


if __name__ == "__main__":
    asyncio.run(run_workflow())
