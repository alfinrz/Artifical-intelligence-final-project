import json # for reading json files

class ObjectDetection: # class for storing object detection data
    def __init__(self, id, position):
        self.id = id
        self.position = position

    @staticmethod
    def from_json(json_obj):
        return ObjectDetection(json_obj['id'], json_obj['position'])

class DetectionData: # class for storing detection data
    def __init__(self, timestamp, size, object_list):
        self.timestamp = timestamp
        self.size = size
        self.object_list = object_list

    @staticmethod
    def from_json(json_detection):
        timestamp = json_detection['timestamp']
        size = json_detection['size']
        object_list = [ObjectDetection.from_json(obj) for obj in json_detection['object_list']]
        return DetectionData(timestamp, size, object_list)

    def objects(self): 
        return self.object_list

    def __len__(self):
        return len(self.object_list)

class TrajectoryData: # class for storing trajectory data
    def __init__(self, id, start_time, positions=None):
        self.id = id
        self.start_time = start_time
        self.positions = positions if positions else []

    def add_position(self, position):
        self.positions.append(position)

    def __len__(self):
        return len(self.positions)

class SampleData: # class for storing sample data
    def __init__(self, id, start_time, positions=None):
        self.trajectory = TrajectoryData(id, start_time, positions)

    @classmethod
    def from_trajectory(cls, trajectory):
        return cls(trajectory.id, trajectory.start_time, positions=trajectory.positions)

    def add_position(self, position):
        self.trajectory.add_position(position)

    @property
    def positions(self):
        return self.trajectory.positions

    @property
    def id(self):
        return self.trajectory.id

    def __len__(self):
        return len(self.trajectory)

    def slice(self, sequence_length, min_length):
        if len(self) < min_length:
            return []

        sliced_samples = []
        start_index = 0
        end_index = sequence_length
        while start_index < len(self):
            new_positions = self.positions[start_index:end_index]
            new_trajectory = TrajectoryData(self.id, self.trajectory.start_time + start_index, positions=new_positions)
            new_sample = SampleData.from_trajectory(new_trajectory)
            start_index += 1
            end_index = start_index + sequence_length

            if len(new_sample) < min_length:
                continue

            sliced_samples.append(new_sample)

        return sliced_samples