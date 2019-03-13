#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.fluid.proto import trainer_desc_pb2
from distributed import ps_pb2 as ps_pb2
from device_worker import DeviceWorkerFactory
from google.protobuf import text_format

__all__ = ['TrainerDesc', 'MultiTrainer', 'DistMultiTrainer']


# can be initialized from train_desc,
class TrainerDesc(object):
    def __init__(self):
        '''
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        '''
        self.proto_desc = trainer_desc_pb2.TrainerDesc()
        import multiprocessing as mp
        # set default thread num == cpu count
        self.proto_desc.thread_num = mp.cpu_count()
        self.fleet_desc_ = None
        self.device_worker_ = None
        self.program_ = None

    def set_thread(self, thread_num):
        self.proto_desc.thread_num = thread_num

    def set_device_worker(self, device_worker):
        self.device_worker_ = device_worker

    def set_fleet_desc(self, fleet_desc):
        self.fleet_desc_ = fleet_desc

    def gen_trainer_desc(self):
        pass

    def set_program(self, program):
        self.program_ = program

    def _desc(self):
        return text_format.MessageToString(self.proto_desc)


class MultiTrainer(TrainerDesc):
    def __init__(self):
        super(MultiTrainer, self).__init__()
        pass

    def gen_trainer_desc(self):
        super(MultiTrainer, self).gen_trainer_desc()
        self.proto_desc.class_name = "MultiTrainer"
        self.device_worker_.gen_worker_desc(self.proto_desc)


class DistMultiTrainer(TrainerDesc):
    def __init__(self):
        super(DistMultiTrainer, self).__init__()
        pass

    def gen_trainer_desc(self):
        super(DistMultiTrainer, self).gen_trainer_desc()
        self.proto_desc.class_name = "DistMultiTrainer"
        self.device_worker_.set_program(self.program_)
        self.device_worker_.gen_worker_desc(self.proto_desc)
