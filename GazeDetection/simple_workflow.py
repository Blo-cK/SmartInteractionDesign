"""Workflow: Face Extractor → NATS → Gaze → Kafka"""
import asyncio


async def run_workflow():
    from GazeDetection.face_extractor_producer import run_producer
    from GazeDetection.simple_consumer_gaze import run_consumer
    
    print("🚀 Pipeline: Face Extractor → NATS → Gaze → Kafka (10 frames)")
    
    await asyncio.gather(
        run_producer(-1),
        run_consumer()
    )
    
    print("\n✅ Pipeline Complete!")


if __name__ == "__main__":
    asyncio.run(run_workflow())
