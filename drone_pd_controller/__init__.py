"""
Drone PD Controller Package

Provides simple PD-based position control for drone waypoint navigation.

This package is separate from policy-based control and includes:
- SimplePDController: Position tracking controller
- waypoint_nav: Main navigation script with define_waypoints, get_path, fly, render_and_save

Usage:
    from drone_pd_controller import SimplePDController
    from drone_pd_controller.waypoint_nav import main
"""

from .pd_controller import SimplePDController
