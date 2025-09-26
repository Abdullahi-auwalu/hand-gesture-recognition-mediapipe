#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # Use the TF Lite Interpreter (works with TF 2.15)
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cache tensor indices for speed
        self._in_idx = self.input_details[0]['index']
        self._out_idx = self.output_details[0]['index']

    # Backward-compatible: returns only the most likely class id (int)
    def __call__(self, landmark_list):
        class_id, _, _ = self.predict(landmark_list)
        return class_id

    # Returns (class_id, confidence, probs)
    def predict(self, landmark_list):
        """Run inference and return class id, confidence and full probs."""
        # Ensure shape [1, D] float32
        x = np.asarray([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(self._in_idx, x)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._out_idx)  # shape [1, C]

        # Some exported models already output probs; if not, softmax them
        probs = self._safe_softmax(logits[0])
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return class_id, confidence, probs

    # For convenience, if you just want the probability vector
    def predict_proba(self, landmark_list):
        _, _, probs = self.predict(landmark_list)
        return probs

    @staticmethod
    def _safe_softmax(logits_row):
        # Numerically-stable softmax (works whether logits or already probs)
        row = np.asarray(logits_row, dtype=np.float32)
        # If the values already sum ~1 and are in [0,1], assume probs
        if np.all(row >= 0) and np.all(row <= 1):
            s = float(row.sum())
            if 0.99 <= s <= 1.01:
                return row
        m = np.max(row)
        ex = np.exp(row - m)
        return ex / np.sum(ex)



# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import numpy as np
# import tensorflow as tf


# class KeyPointClassifier(object):
#     def __init__(
#         self,
#         model_path='model/keypoint_classifier/keypoint_classifier.tflite',
#         num_threads=1,
#     ):
#         self.interpreter = tf.lite.Interpreter(model_path=model_path,
#                                                num_threads=num_threads)

#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#     def __call__(
#         self,
#         landmark_list,
#     ):
#         input_details_tensor_index = self.input_details[0]['index']
#         self.interpreter.set_tensor(
#             input_details_tensor_index,
#             np.array([landmark_list], dtype=np.float32))
#         self.interpreter.invoke()

#         output_details_tensor_index = self.output_details[0]['index']

#         result = self.interpreter.get_tensor(output_details_tensor_index)

#         result_index = np.argmax(np.squeeze(result))

#         return result_index
