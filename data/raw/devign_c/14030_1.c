static int nvdec_vp8_start_frame(AVCodecContext *avctx, const uint8_t *buffer, uint32_t size)

{

    VP8Context *h = avctx->priv_data;



    NVDECContext      *ctx = avctx->internal->hwaccel_priv_data;

    CUVIDPICPARAMS     *pp = &ctx->pic_params;

    FrameDecodeData *fdd;

    NVDECFrame *cf;

    AVFrame *cur_frame = h->framep[VP56_FRAME_CURRENT]->tf.f;



    int ret;



    ret = ff_nvdec_start_frame(avctx, cur_frame);

    if (ret < 0)

        return ret;



    fdd = (FrameDecodeData*)cur_frame->private_ref->data;

    cf  = (NVDECFrame*)fdd->hwaccel_priv;



    *pp = (CUVIDPICPARAMS) {

        .PicWidthInMbs     = (cur_frame->width  + 15) / 16,

        .FrameHeightInMbs  = (cur_frame->height + 15) / 16,

        .CurrPicIdx        = cf->idx,



        .CodecSpecific.vp8 = {

            .width                       = cur_frame->width,

            .height                      = cur_frame->height,



            .first_partition_size        = h->header_partition_size,



            .LastRefIdx                  = safe_get_ref_idx(h->framep[VP56_FRAME_PREVIOUS]),

            .GoldenRefIdx                = safe_get_ref_idx(h->framep[VP56_FRAME_GOLDEN]),

            .AltRefIdx                   = safe_get_ref_idx(h->framep[VP56_FRAME_GOLDEN2]),



            .frame_type                  = !h->keyframe,

            .version                     = h->profile,

            .show_frame                  = !h->invisible,

            .update_mb_segmentation_data = h->segmentation.enabled ? h->segmentation.update_feature_data : 0,

       }

    };



    return 0;

}
