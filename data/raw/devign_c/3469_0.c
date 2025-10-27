static int nvdec_vc1_start_frame(AVCodecContext *avctx, const uint8_t *buffer, uint32_t size)

{

    VC1Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;



    NVDECContext      *ctx = avctx->internal->hwaccel_priv_data;

    CUVIDPICPARAMS     *pp = &ctx->pic_params;

    FrameDecodeData *fdd;

    NVDECFrame *cf;

    AVFrame *cur_frame = s->current_picture.f;



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

        .field_pic_flag    = v->field_mode,

        .bottom_field_flag = v->cur_field_type,

        .second_field      = v->second_field,



        .intra_pic_flag    = s->pict_type == AV_PICTURE_TYPE_I ||

                             s->pict_type == AV_PICTURE_TYPE_BI,

        .ref_pic_flag      = s->pict_type == AV_PICTURE_TYPE_I ||

                             s->pict_type == AV_PICTURE_TYPE_P,



        .CodecSpecific.vc1 = {

            .ForwardRefIdx     = get_ref_idx(s->last_picture.f),

            .BackwardRefIdx    = get_ref_idx(s->next_picture.f),

            .FrameWidth        = cur_frame->width,

            .FrameHeight       = cur_frame->height,



            .intra_pic_flag    = s->pict_type == AV_PICTURE_TYPE_I ||

                                 s->pict_type == AV_PICTURE_TYPE_BI,

            .ref_pic_flag      = s->pict_type == AV_PICTURE_TYPE_I ||

                                 s->pict_type == AV_PICTURE_TYPE_P,

            .progressive_fcm   = v->fcm == 0,



            .profile           = v->profile,

            .postprocflag      = v->postprocflag,

            .pulldown          = v->broadcast,

            .interlace         = v->interlace,

            .tfcntrflag        = v->tfcntrflag,

            .finterpflag       = v->finterpflag,

            .psf               = v->psf,

            .multires          = v->multires,

            .syncmarker        = v->resync_marker,

            .rangered          = v->rangered,

            .maxbframes        = s->max_b_frames,



            .panscan_flag      = v->panscanflag,

            .refdist_flag      = v->refdist_flag,

            .extended_mv       = v->extended_mv,

            .dquant            = v->dquant,

            .vstransform       = v->vstransform,

            .loopfilter        = v->s.loop_filter,

            .fastuvmc          = v->fastuvmc,

            .overlap           = v->overlap,

            .quantizer         = v->quantizer_mode,

            .extended_dmv      = v->extended_dmv,

            .range_mapy_flag   = v->range_mapy_flag,

            .range_mapy        = v->range_mapy,

            .range_mapuv_flag  = v->range_mapuv_flag,

            .range_mapuv       = v->range_mapuv,

            .rangeredfrm       = v->rangeredfrm,

        }

    };



    return 0;

}
