from backend.backend import OpenAICompitableProxyBackendBase
from emd.utils.logger_utils import get_logger
import glob
import os
import time
import threading

logger = get_logger(__name__)

class KTransformersBackend(OpenAICompitableProxyBackendBase):
    server_port = "10002"

    def find_gguf_file(self,model_path):
        if os.path.exists(model_path):
            if model_path.endswith(".gguf"):
                return model_path
            else:
                gghf_file_paths = glob.glob(os.path.join(model_path,"**/*.gguf"),recursive=True)
                if not gghf_file_paths:
                    raise ValueError(f"no gguf file found in {model_path}")

                if  len(gghf_file_paths) == 1:
                    return gghf_file_paths[0]

                first_gguf_file_paths = glob.glob(os.path.join(model_path,"**/*001*.gguf"),recursive=True)
                if not first_gguf_file_paths:
                    raise ValueError(f"no 001 gguf file found in {model_path}, all gguf files: {gghf_file_paths}")
                if len(first_gguf_file_paths) > 1:
                    raise ValueError(f"multiple 001 gguf files found in {model_path}, all gguf files: {gghf_file_paths}")
                return first_gguf_file_paths[0]
        else:
            raise FileNotFoundError(f"model path {model_path} not found")


    def format_devices(self):
        return ",".join([f"CUDA{i}" for i in range(self.gpu_num)])

    def create_proxy_server_start_command(self,model_path):
        # find gguf from model path
        gguf_model_path = self.find_gguf_file(model_path)

        TORCH_CUDA_ARCH_LIST = None
        if 'g5' in self.instance_type:
            TORCH_CUDA_ARCH_LIST = '8.6'
        elif 'g6' in self.instance_type:
            TORCH_CUDA_ARCH_LIST = '8.9'
        else:
            raise ValueError(f"Unsupported instance type!")

        cpu_infer = self.cpu_num - 2

        serve_command = f'TORCH_CUDA_ARCH_LIST={TORCH_CUDA_ARCH_LIST} python /opt/ml/code/ktransformers/ktransformers/server/main.py  --model_path /opt/ml/model/DeepSeek-R1  --gguf_path {gguf_model_path} --port {self.server_port} --cpu_infer {cpu_infer}'
        if self.environment_variables:
            serve_command = f'{self.environment_variables} && {serve_command}'
        return serve_command

    def invoke(self, request):
        # Transform input to lmdeploy format
        request = self._transform_request(request)
        logger.info(f"Chat request:{request}")
        response = self.client.chat.completions.create(**request)
        logger.info(f"response:{response}")
        if request.get('stream',False):
            return self._transform_streaming_response(response)
        else:
            return self._transform_response(response)
