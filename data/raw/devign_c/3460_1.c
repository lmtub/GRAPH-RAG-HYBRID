static int vdpau_mpeg_start_frame(AVCodecContext *avctx,

                                  const uint8_t *buffer, uint32_t size)

{

    MpegEncContext * const s = avctx->priv_data;

    Picture *pic             = s->current_picture_ptr;

    struct vdpau_picture_context *pic_ctx = pic->hwaccel_picture_private;

    VdpPictureInfoMPEG1Or2 *info = &pic_ctx->info.mpeg;

    VdpVideoSurface ref;

    int i;



    /* fill VdpPictureInfoMPEG1Or2 struct */

    info->forward_reference  = VDP_INVALID_HANDLE;

    info->backward_reference = VDP_INVALID_HANDLE;



    switch (s->pict_type) {

    case AV_PICTURE_TYPE_B:

        ref = ff_vdpau_get_surface_id(&s->next_picture.f);

        assert(ref != VDP_INVALID_HANDLE);

        info->backward_reference = ref;

        /* fall through to forward prediction */

    case AV_PICTURE_TYPE_P:

        ref = ff_vdpau_get_surface_id(&s->last_picture.f);

        info->forward_reference  = ref;

    }



    info->slice_count                = 0;

    info->picture_structure          = s->picture_structure;

    info->picture_coding_type        = s->pict_type;

    info->intra_dc_precision         = s->intra_dc_precision;

    info->frame_pred_frame_dct       = s->frame_pred_frame_dct;

    info->concealment_motion_vectors = s->concealment_motion_vectors;

    info->intra_vlc_format           = s->intra_vlc_format;

    info->alternate_scan             = s->alternate_scan;

    info->q_scale_type               = s->q_scale_type;

    info->top_field_first            = s->top_field_first;

    // Both for MPEG-1 only, zero for MPEG-2:

    info->full_pel_forward_vector    = s->full_pel[0];

    info->full_pel_backward_vector   = s->full_pel[1];

    // For MPEG-1 fill both horizontal & vertical:

    info->f_code[0][0]               = s->mpeg_f_code[0][0];

    info->f_code[0][1]               = s->mpeg_f_code[0][1];

    info->f_code[1][0]               = s->mpeg_f_code[1][0];

    info->f_code[1][1]               = s->mpeg_f_code[1][1];

    for (i = 0; i < 64; ++i) {

        info->intra_quantizer_matrix[i]     = s->intra_matrix[i];

        info->non_intra_quantizer_matrix[i] = s->inter_matrix[i];

    }



    return ff_vdpau_common_start_frame(pic_ctx, buffer, size);

}
