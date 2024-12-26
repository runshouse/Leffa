from predict import LeffaPredictor

# Initialize once
predictor = LeffaPredictor()

# Virtual try-on
result = predictor.virtual_tryon(
    person_image_path="person.jpg",
    garment_image_path="jersey.jpg",
    garment_type="upper_body"  # or "lower_body" or "dresses"
)
result.save("result.png")

# # Pose transfer
# result = predictor.pose_transfer(
#     source_image_path="path/to/source.jpg",
#     target_pose_image_path="path/to/target_pose.jpg"
# )
# result.save("pose_transfer_result.png")