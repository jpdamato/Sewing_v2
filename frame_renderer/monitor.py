import glfw



class MonitorData:
    def __init__(self, monitor):
        self.name = glfw.get_monitor_name(monitor)
        self.position = glfw.get_monitor_pos(monitor)
        mode = glfw.get_video_mode(monitor)
        self.width = mode.size.width
        self.height = mode.size.height
        self.refresh_rate = mode.refresh_rate
    
    def __str__(self):
        return (f"MonitorData(name={self.name!r}, "
                f"position={self.position}, "
                f"size=({self.width}, {self.height}), "
                f"refresh_rate={self.refresh_rate}Hz)")
    
