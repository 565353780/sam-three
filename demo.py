from sam_three.Demo.detector import demo as demo_detect


if __name__ == '__main__':
    demo_detect()

'''
IMG_WIDTH, IMG_HEIGHT = image_list[0].size

points_tensor = torch.tensor(
    [
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)
# positive clicks have label 1, while negative clicks have label 0
points_labels_tensor = torch.tensor(
    [1],
    dtype=torch.int32,
)

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=1,
        points=points_tensor,
        point_labels=points_labels_tensor,
        obj_id=0,
    )
)
out = response["outputs"]

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and plot the outputs every 60 frames
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
'''
