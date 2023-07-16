import sys
sys.path.append("silk/")
from silk_detector import SiLKDetector
from silk_descriptor import SiLKDescriptor
from silk_matcher import DualSoftMaxMatcher

from mega1500 import MegaDepthPoseMNNBenchmark


def test():
    detector = SiLKDetector()
    descriptor = SiLKDescriptor()
    mega_pose_bench = MegaDepthPoseMNNBenchmark()
    mega_pose_bench.benchmark(detector_model = detector, 
                            descriptor_model = descriptor, 
                            matcher_model = DualSoftMaxMatcher())
if __name__ == "__main__":
    test()
    
    
