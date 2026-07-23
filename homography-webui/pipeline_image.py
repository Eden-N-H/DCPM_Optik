delta_heading = math.radians(view_heading - base_heading)
cos_d, sin_d = math.cos(delta_heading), math.sin(delta_heading)
cam_offset_x = x_base * cos_d + z_base * sin_d
cam_offset_z = -x_base * sin_d + z_base * cos_d
