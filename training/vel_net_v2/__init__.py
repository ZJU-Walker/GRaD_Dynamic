"""
Training pipeline for VelNetV2.

Multi-stage training:
  Stage 1: Geometry branch (angular velocity + translation direction)
  Stage 2: Dynamics branch (scale + correction), geometry frozen
  Stage 3: Joint fine-tuning (all branches)
"""
