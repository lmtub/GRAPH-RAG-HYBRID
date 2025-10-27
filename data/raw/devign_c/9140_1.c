static void libschroedinger_handle_first_access_unit(AVCodecContext *avctx)

{

    SchroDecoderParams *p_schro_params = avctx->priv_data;

    SchroDecoder *decoder = p_schro_params->decoder;



    p_schro_params->format = schro_decoder_get_video_format(decoder);



    /* Tell FFmpeg about sequence details. */

    if (av_image_check_size(p_schro_params->format->width,

                            p_schro_params->format->height, 0, avctx) < 0) {

        av_log(avctx, AV_LOG_ERROR, "invalid dimensions (%dx%d)\n",

               p_schro_params->format->width, p_schro_params->format->height);

        avctx->height = avctx->width = 0;

        return;

    }

    avctx->height  = p_schro_params->format->height;

    avctx->width   = p_schro_params->format->width;

    avctx->pix_fmt = get_chroma_format(p_schro_params->format->chroma_format);



    if (ff_get_schro_frame_format(p_schro_params->format->chroma_format,

                                  &p_schro_params->frame_format) == -1) {

        av_log(avctx, AV_LOG_ERROR,

               "This codec currently only supports planar YUV 4:2:0, 4:2:2 "

               "and 4:4:4 formats.\n");

        return;

    }



    avctx->framerate.num = p_schro_params->format->frame_rate_numerator;

    avctx->framerate.den = p_schro_params->format->frame_rate_denominator;

}
