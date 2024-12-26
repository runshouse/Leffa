import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


class LeffaPredictor:
    def __init__(self, download_weights=True):
        # Download checkpoints if needed
        if download_weights:
            snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

        # Initialize models
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        # Initialize virtual try-on model
        vt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
        )
        self.vt_inference = LeffaInference(model=vt_model)
        self.vt_model_type = "viton_hd"

        # Initialize pose transfer model
        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def virtual_tryon(self, person_image_path, garment_image_path, garment_type="upper_body", steps=50, scale=2.5, seed=42):
        """
        Perform virtual try-on: Put garment on person
        Args:
            person_image_path (str): Path to person image
            garment_image_path (str): Path to garment image
            garment_type (str): Type of garment - "upper_body", "lower_body", or "dresses"
            steps (int): Number of inference steps
            scale (float): Guidance scale
            seed (int): Random seed
        Returns:
            PIL.Image: Generated image with garment tried on
        """
        output = self.leffa_predict(
            person_image_path, 
            garment_image_path,
            "virtual_tryon",
            step=steps,
            scale=scale,
            seed=seed,
            garment_type=garment_type
        )
        return Image.fromarray(output)

    def pose_transfer(self, source_image_path, target_pose_image_path, steps=50, scale=2.5, seed=42):
        """
        Perform pose transfer: Transfer source person to target pose
        Args:
            source_image_path (str): Path to source person image
            target_pose_image_path (str): Path to target pose image
            steps (int): Number of inference steps
            scale (float): Guidance scale
            seed (int): Random seed
        Returns:
            PIL.Image: Generated image with transferred pose
        """
        output = self.leffa_predict(
            source_image_path,
            target_pose_image_path,
            "pose_transfer",
            step=steps,
            scale=scale,
            seed=seed
        )
        return Image.fromarray(output)

    def leffa_predict(self, src_image_path, ref_image_path, control_type, step=50, scale=2.5, seed=42, garment_type="upper_body"):
        assert control_type in [
            "virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
        
        # Load and preprocess images
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)
        src_image_array = np.array(src_image)

        # Generate mask
        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            if self.vt_model_type == "viton_hd":
                garment_type_hd = "upper" if garment_type in [
                    "upper_body", "dresses"] else "lower"
                mask = self.mask_predictor(src_image, garment_type_hd)["mask"]
            elif self.vt_model_type == "dress_code":
                keypoints = self.openpose(src_image.resize((384, 512)))
                model_parse, _ = self.parsing(src_image.resize((384, 512)))
                mask = get_agnostic_mask(model_parse, keypoints, garment_type)
                mask = mask.resize((768, 1024))
        else:  # pose_transfer
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        # Generate DensePose
        if control_type == "virtual_tryon":
            if self.vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)
                densepose = Image.fromarray(src_image_seg_array)
            elif self.vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                densepose = Image.fromarray(src_image_seg_array)
        else:  # pose_transfer
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
            densepose = Image.fromarray(src_image_iuv_array)

        # Prepare data for model
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        # Run inference
        inference = self.vt_inference if control_type == "virtual_tryon" else self.pt_inference
        output = inference(
            data,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
        )
        
        return np.array(output["generated_image"][0])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Leffa Virtual Try-On and Pose Transfer")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--garment", required=True, help="Path to garment image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    parser.add_argument("--garment-type", default="upper_body", 
                      choices=["upper_body", "lower_body", "dresses"],
                      help="Type of garment")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--scale", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # Initialize predictor
    predictor = LeffaPredictor()

    # Perform virtual try-on
    result = predictor.virtual_tryon(
        args.person,
        args.garment,
        garment_type=args.garment_type,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed
    )

    # Save result
    result.save(args.output)
    print(f"Result saved to {args.output}")