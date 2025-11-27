from  architecture.library.output_layer_monitor import OutputLayerMonitor


if __name__ == "__main__":
    print("Starting monitor...")
    monitor = OutputLayerMonitor(
        source_name="camera1",
        service="object_detection"
    )
    monitor.start(flask_port=5000)
