import logging
import time
import threading
import queue
from typing import Dict, Optional, Tuple, Union, Any
import copy
from typing_extensions import override
import websockets.sync.client
import websockets.exceptions
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

import sys
import io
import torch
import torchvision.transforms as T
from PIL import Image
import os


class WebsocketClientPolicy(_base_policy.BasePolicy):
    
    _STREAM_END_SENTINEL = object()
    
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._action_queue: queue.Queue = queue.Queue(maxsize=100)
        self._receiver_thread: Optional[threading.Thread] = None
        self._stream_active: bool = True 
        self._lock = threading.Lock()
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        _, self._server_metadata = self._wait_for_metadata_only()
        self._receiver_thread = threading.Thread(target=self._stream_receiver_loop, daemon=True)
        self._receiver_thread.start()       
        time.sleep(0.1) 


    def get_server_metadata(self) -> Dict:
        return self._server_metadata
    


    def _wait_for_metadata_only(self) -> Tuple[None, Dict]:
        logging.info(f"Waiting for server at {self._uri} for metadata...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None 
                with websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                ) as conn:
                   
                    metadata = msgpack_numpy.unpackb(conn.recv())
                    logging.info("Metadata successfully received.")
                    return None, metadata
                
            except ConnectionRefusedError:
                
                logging.info("Still waiting for server...")
                time.sleep(5)
            except websockets.exceptions.ConnectionClosedOK:
                
                logging.info("Metadata connection closed gracefully, retrying to connect...")
                time.sleep(5)
            except websockets.exceptions.ConnectionClosedError as e:
                
                logging.warning(f"Metadata connection closed unexpectedly: {e}. Retrying...")
                time.sleep(5)
            except Exception as e:
                
                logging.warning(f"Connection attempt failed with error: {e}. Retrying...")
                time.sleep(5)

    def _ensure_connection(self) -> bool:
        if self._ws is not None:
            return True 

        logging.info(f"Establishing new connection to {self._uri}...")
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, additional_headers=headers
            )
            
            try:
                _ = self._ws.recv(timeout=1.0) 
            except Exception:
                pass 
                
            logging.info("Connection established and initial message/metadata received.")
            return True
        except Exception as e:
            logging.error(f"Failed to establish connection: {e}", exc_info=True)
            self._ws = None 
            return False

  
    def get_left_queue_actions(self) -> np.ndarray:
      
        with self._action_queue.mutex:
            if not self._action_queue.queue:
                print("Queue is empty, returning inf.")
                return np.zeros(7, dtype=np.float32)
            queue_snapshot = list(self._action_queue.queue)
        summed_action = None
        
        for element_dict in queue_snapshot:
            if isinstance(element_dict, dict) and "actions" in element_dict:
                try:
                    action_np = np.asarray(element_dict["actions"], dtype=np.float32).flatten()
                except (ValueError, TypeError):
                    continue

                if summed_action is None:
                    summed_action = np.zeros_like(action_np)

                curr_size = action_np.size
                acc_size = summed_action.size

                if curr_size > acc_size:
                    new_sum = np.zeros(curr_size, dtype=np.float32)
                    new_sum[:acc_size] = summed_action
                    summed_action = new_sum

                summed_action[:curr_size] += action_np

        if summed_action is None:
            return np.zeros(7, dtype=np.float32)

        return summed_action[:7].astype(np.float32)



    def _stream_receiver_loop(self) -> None:
        logging.info("[Receiver Thread] Stream thread started and running continuously.")
        
        while self._stream_active:
            if not self._ensure_connection():
                continue

            try:
                response = self._ws.recv() 
                
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")
                unpacked_response = msgpack_numpy.unpackb(response)

                if self._action_queue.full():
                    logging.info(f"Queue Is Full !")
                self._action_queue.put(unpacked_response)
                logging.debug(f"[Receiver Thread] Action received. Size: {self._action_queue.qsize()}")

            except websockets.exceptions.ConnectionClosed:
                logging.warning("[Receiver Thread] Connection closed. Attempting to re-establish.")
                with self._lock:
                     if self._ws:
                          try:
                               self._ws.close()
                          except Exception:
                               pass
                     self._ws = None 
                time.sleep(1) 
            except Exception as e:
                if self._stream_active: 
                    logging.error(f"[Receiver Thread] Error during stream reception: {e}", exc_info=True)

                with self._lock:
                     self._ws = None 
              
        logging.info("[Receiver Thread] Exiting due to stream_active=False.")


    @override
    def infer(self, obs: Dict,new_task: bool) -> None:  
        # if it is a new task, we clear the queue to avoid executing old actions from the previous task.
        if new_task:
            with self._action_queue.mutex:
                self._action_queue.queue.clear() 

        logging.info(f"[Client] : queue empty: {self._action_queue.empty()}")
        
        with self._lock:
            if not self._ensure_connection():
                logging.warning("[client] Cannot send observation: Connection is down.")
                return 

            try:
                
                data = self._packer.pack(obs)
                self._ws.send(data) 

            except websockets.exceptions.ConnectionClosed:
                
                logging.warning("Connection closed during send. _Receiver_loop will handle re-establishment.")
                self._ws = None
            except Exception as e:
                logging.error(f"Unexpected error during send: {e}")
                self._ws = None
                return
                
    def get_queue_length(self) -> int:
        return self._action_queue.qsize()

    def get_next_action(self, timeout: Optional[float] = 5) -> Union[Dict, None]:
        with self._lock:
            
            try:
                if not self._action_queue.empty():
                    item = self._action_queue.get(timeout=timeout)      
                    if item is not None and 'actions' in item:
                        actions = item['actions']
                
                        if actions.ndim == 2 and actions.shape[0] == 1:
                            item['actions'] = actions.flatten() 
                        elif actions.ndim != 1:
                            logging.warning(f"Action array has unexpected shape: {actions.shape}. Environment may fail.")
        
                    self._action_queue.task_done()
                    return item
                else:
                    return None
            except Exception as e:
                logging.error(f"Error getting action from queue: {e}", exc_info=True)
                return None