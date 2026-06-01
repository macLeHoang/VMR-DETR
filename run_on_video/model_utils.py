import torch
from vmr_detr.modeling.transformer import build_transformer
from vmr_detr.modeling.position_encoding import build_position_encoding
from vmr_detr.modeling.model import VMRDETR


def build_inference_model(ckpt_path, **kwargs):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["opt"]
    if len(kwargs) > 0:  # used to overwrite default args
        args.update(kwargs)
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = VMRDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        use_temporal_pyramid=getattr(args, "use_temporal_pyramid", False),
        temporal_pyramid_downsample=getattr(args, "temporal_pyramid_downsample", "avg"),
    )

    model.load_state_dict(ckpt["model"])
    return model
