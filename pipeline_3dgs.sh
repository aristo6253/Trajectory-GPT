set -e  # exit on any error

EXP_NAME='testV'
TRAJ='corner_chairV'
model='output/garden_test'
TRAJ_DESCRIPTION="Reach the chair in the corner of the room by flying over the table making sure not to collide with the table or the chairs in the middle of the scene. You should keep the target chair always in frame and stop when the target chair is the center of the view."


# Provide starting position, having a json with one entry with (R, T, FoVx, FoVy, width, height, id=step)
python render.py -m ${model} --my_traj --trajectory_file ${TRAJ}.json

# Generate depth and bev

# Provide everything to the gpt

# Transform the gpt output to an extrinsic matrix (R and T), follow ViewCrafter trajectory creation method