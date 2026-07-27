"""Honest locate_object_3d perception package for SO-101.

Modules:
  camera_math      shared FK camera geometry + ray triangulation (numpy only)
  detector         FastSAM everything-mode + color-free on-table selector (torch)
  sidecar          persistent inference server (scratch venv)
  client           robot-venv client that drives the sidecar over stdio
  locate_object_3d orchestrator: images + FK poses -> (x,y,z,confidence,method)
"""
