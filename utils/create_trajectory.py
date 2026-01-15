#!/usr/bin/env python3
"""
Create a trajectory CSV file for an object's movement.
The object moves back and forth between two points (A and B) in a continuous
loop at a constant speed.
"""

import pandas as pd
import numpy as np
import os

def create_looping_trajectory():
    """
    Generates a trajectory that oscillates between two points at a constant speed.
    The output is saved as a CSV file.
    """
    
    # --- Parameters to Modify ---
    
    # The two points for the object to move between.ß
    point_a = np.array([3.0, 0, 0.3])
    point_b = np.array([3.0, 1, 0.3])
    
    # The constant speed of the object in m/s.
    speed = 0.5  # m/s
    
    # Total duration of the trajectory in seconds.
    total_duration = 50.0
    
    # Time step for simulation in seconds.
    dt = 0.01
    
    # --- Trajectory Calculation ---
    
    # Calculate the distance and vector for a one-way trip
    vector_ab = point_b - point_a
    distance = np.linalg.norm(vector_ab)
    
    # Handle edge cases where speed is zero or points are the same
    if speed <= 0:
        print("⚠️ Speed must be positive. Defaulting to 0.1 m/s.")
        speed = 0.1
    
    if distance == 0:
        print("⚠️ Point A and Point B are the same. The object will not move.")
        one_way_duration = float('inf')
    else:
        one_way_duration = distance / speed

    cycle_duration = 2 * one_way_duration # Time for A -> B -> A
    
    # Calculate the total number of points for the trajectory
    num_points = int(total_duration / dt) + 1
    
    # Create the time array
    times = np.linspace(0, total_duration, num_points)
    
    # Initialize position arrays
    positions_x = np.zeros(num_points)
    positions_y = np.zeros(num_points)
    positions_z = np.zeros(num_points)
    
    # Calculate positions at each time step
    for i, t in enumerate(times):
        if distance == 0:
            current_pos = point_a
        else:
            # Determine where in the A -> B -> A cycle we are
            time_in_cycle = t % cycle_duration
            
            if time_in_cycle <= one_way_duration:
                # Moving from A to B
                fraction = time_in_cycle / one_way_duration
                current_pos = point_a + vector_ab * fraction
            else:
                # Moving from B back to A
                time_in_return = time_in_cycle - one_way_duration
                fraction = time_in_return / one_way_duration
                # Start at B and move in the opposite direction of vector_ab
                current_pos = point_b - vector_ab * fraction
        
        positions_x[i] = current_pos[0]
        positions_y[i] = current_pos[1]
        positions_z[i] = current_pos[2]
            
    # Quaternions for orientation (identity quaternion means no rotation)
    quat_x = np.zeros(num_points)
    quat_y = np.zeros(num_points)
    quat_z = np.zeros(num_points)
    quat_w = np.ones(num_points)
    
    # Create a pandas DataFrame to store the trajectory data
    df = pd.DataFrame({
        'time': times,
        'position_x': positions_x,
        'position_y': positions_y,
        'position_z': positions_z,
        'quat_x': quat_x,
        'quat_y': quat_y,
        'quat_z': quat_z,
        'quat_w': quat_w
    })
    
    # --- File Output and Logging ---
    
    output_dir = '/home/irislab/ke/GRaD_Dynamic_onboard/envs/assets/trajectories'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'human.csv')
    df.to_csv(output_file, index=False, float_format='%.4f')
    
    # --- Print Summary to Console ---
    
    print(f"✅ Looping trajectory file created: {output_file}")
    print("\n--- Trajectory Details ---")
    print(f"  - Point A:                 {point_a.tolist()}")
    print(f"  - Point B:                 {point_b.tolist()}")
    print(f"  - Speed:                   {speed} m/s")
    print(f"  - One-way Trip Duration:   {one_way_duration:.2f} seconds")
    print(f"  - Full Cycle (A->B->A):    {cycle_duration:.2f} seconds")
    print(f"  - Total Duration:          {total_duration} seconds")
    print(f"  - Total Points:            {num_points}")
    
    # Print some sample points for verification
    print("\n--- Sample Trajectory Points ---")
    print("Time(s) | X(m)   | Y(m)   | Z(m)")
    print("-" * 40)
    
    # Define indices for sampling at key moments
    sample_indices = [0, 1, 2, 3, 4] # Start
    if distance > 0:
        # Mid-point of A->B
        sample_indices.append(int(one_way_duration / 2 / dt))
        # End of A->B (at Point B)
        sample_indices.append(int(one_way_duration / dt))
        # Mid-point of B->A
        sample_indices.append(int((one_way_duration * 1.5) / dt))
    sample_indices.extend(range(len(times)-5, len(times))) # End
    
    # Remove duplicates and sort
    sample_indices = sorted(list(set(sample_indices)))
    
    for i in sample_indices:
        if i < len(times):
            t = times[i]
            x = positions_x[i]
            y = positions_y[i]
            z = positions_z[i]
            print(f"{t:7.2f} | {x:6.2f} | {y:6.2f} | {z:6.2f}")
    
    return df

if __name__ == "__main__":
    create_looping_trajectory()