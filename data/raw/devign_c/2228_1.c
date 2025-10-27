static unsigned int get_video_format_idx(AVCodecContext *avctx)

{

    unsigned int ret_idx = 0;

    unsigned int idx;

    unsigned int num_formats = sizeof(ff_schro_video_format_info) /

                               sizeof(ff_schro_video_format_info[0]);



    for (idx = 1; idx < num_formats; ++idx) {

        const SchroVideoFormatInfo *vf = &ff_schro_video_format_info[idx];

        if (avctx->width  == vf->width &&

            avctx->height == vf->height) {

            ret_idx = idx;

            if (avctx->time_base.den == vf->frame_rate_num &&

                avctx->time_base.num == vf->frame_rate_denom)

                return idx;

        }

    }

    return ret_idx;

}
