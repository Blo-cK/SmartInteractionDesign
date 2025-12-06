from dataclasses import dataclass, asdict
import time
from typing import Any, Callable
import json
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .input_layer import InputResultWrapper



class InvalidMetadataError(Exception):
    """Is being thrown if Metadata is missing or corrupt."""
    pass


@dataclass
class OutputLayerMetadata:
    """ source_id = The sensor, for example camera1, microphone1...
        service_id = The service which processed the data e.g. Gaze Detection etc.
        time_stamp = The Timestamp when the sensor input got created
        completed_at = The Timestamp when the service finished processing the data
        result = Result from the Service
    """

    source_id: str
    service_id: str
    time_stamp: str
    completed_at: str
    result: dict

    def to_dict(self):
        return asdict(self)


class OutputLayerProducer:
    def __init__(self, broker: str = "152.53.32.66:9094"):
        self.broker = broker
        self._connected = False
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.broker,
            value_serializer=self._smart_serializer,
            compression_type=None,
            linger_ms=5,
            max_request_size=1024 * 1024,
            acks=0
        )

    def _smart_serializer(self, v):
        if isinstance(v, dict):
            return json.dumps(v).encode("utf-8")

        if isinstance(v, str):
            return v.encode("utf-8")

        return json.dumps(v).encode("utf-8")

    async def _connect(self):
        if not self._connected:
            await self.producer.start()
            self._connected = True
            print(f"[SmartInteraction] Connected OutputLayerProducer to {self.broker}")

    def _build_topic(self, metadata: OutputLayerMetadata) -> str:
        if metadata.source_id and metadata.service_id:
            return f"output.{metadata.source_id}.{metadata.service_id}".lower()
        raise InvalidMetadataError("Source name and service needs to be declared to build a topic!")

    async def sendData(self, input_result : InputResultWrapper, result, service_id : str):
        """Send serialized data to Kafka"""
        if not self._connected:
            await self._connect()
        
        try:
            metadata = self._map_header_to_output_layer_metadata(input_result.msg.headers, result, service_id)
            topic = self._build_topic(metadata)
            print("TOPIC SEND: ", topic)
            data = metadata.to_dict()
            await self.producer.send(topic, data)
            
        except Exception as e:
            print(f"[SmartInteraction] Error sending data: {e}")


    def _map_header_to_output_layer_metadata(self, header: dict[str, Any], result, service_id) -> OutputLayerMetadata:
        required_keys = ["time_stamp", "source_id"]

        missing = [key for key in required_keys if key not in header]
        if missing:
            raise KeyError(f"Missing required header fields: {missing}")

        metadata = OutputLayerMetadata(
            time_stamp=header["time_stamp"],
            service_id=service_id,
            source_id=header["source_id"],
            result=result,
            completed_at=int(time.time())
        )

        return metadata


    async def disconnect(self):
        if self._connected:
            await self.producer.stop()
            self._connected = False
            print("[SmartInteraction] Disconnected OutputLayerProducer")


class OutputLayerReceiver:
    def __init__(self, broker: str = "152.53.32.66:9094", group_id: str = None):
        self.broker = broker
        self.group_id = group_id
        self.consumer = None
        self._connected = False

    async def _connect(self, topic: str):
        if self._connected:
            return

        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.broker,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8"))
        )
        await self.consumer.start()
        self._connected = True
        print(f"[SmartInteraction] Connected OutputLayerReceiver to topic '{topic}'")

    async def receiveData(self, source_name: str, service: str, onData: Callable):
        """Consume data messages and call callback (single-topic mode)"""
        topic = f"output.{source_name}.{service}"

        if not self._connected:
            await self._connect(topic)

        try:
            async for msg in self.consumer:
               
                metadata_dict = msg.value
                try:
                    metadata = OutputLayerMetadata(**metadata_dict)
                    if onData:
                        await onData(metadata)
                except Exception as e:
                    print(f"[SmartInteraction] Error parsing data: {e}")
        finally:
            await self.consumer.stop()
            self._connected = False
            print(f"[SmartInteraction] Disconnected OutputLayerReceiver from topic '{topic}'")


    async def _connect_pattern(self, pattern: str):
        """Connect using a Kafka topic regex pattern."""
        if self._connected:
            return

        self.consumer = AIOKafkaConsumer(
            bootstrap_servers=self.broker,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8"))
        )

        self.consumer.subscribe(pattern=pattern)

        await self.consumer.start()
        self._connected = True
        print(f"[SmartInteraction] Connected OutputLayerReceiver with pattern '{pattern}'")

    async def receiveAllData(self, onData: Callable):
        """Consume data from all topics matching output.*.*"""
        pattern = r"output\..*\..*"
        if not self._connected:
            await self._connect_pattern(pattern)
          
        try:
            async for msg in self.consumer:
                
                metadata_dict = msg.value
                try:
                  
                    metadata = OutputLayerMetadata(**metadata_dict)
                    if onData:
                        await onData(metadata)
                except Exception as e:
                    print(f"[SmartInteraction] Error parsing data: {e}")
        finally:
            await self.consumer.stop()
            self._connected = False
            print("[SmartInteraction] Disconnected OutputLayerReceiver")

    async def disconnect(self):
        if self.consumer and self._connected:
            await self.consumer.stop()
            self._connected = False
            print("[SmartInteraction] Receiver disconnected")
