static av_cold int libschroedinger_encode_init(AVCodecContext *avctx)

{

    SchroEncoderParams *p_schro_params = avctx->priv_data;

    SchroVideoFormatEnum preset;



    /* Initialize the libraries that libschroedinger depends on. */

    schro_init();



    /* Create an encoder object. */

    p_schro_params->encoder = schro_encoder_new();



    if (!p_schro_params->encoder) {

        av_log(avctx, AV_LOG_ERROR,

               "Unrecoverable Error: schro_encoder_new failed. ");

        return -1;

    }



    /* Initialize the format. */

    preset = ff_get_schro_video_format_preset(avctx);

    p_schro_params->format =

                    schro_encoder_get_video_format(p_schro_params->encoder);

    schro_video_format_set_std_video_format(p_schro_params->format, preset);

    p_schro_params->format->width  = avctx->width;

    p_schro_params->format->height = avctx->height;



    if (set_chroma_format(avctx) == -1)

        return -1;



    if (avctx->color_primaries == AVCOL_PRI_BT709) {

        p_schro_params->format->colour_primaries = SCHRO_COLOUR_PRIMARY_HDTV;

    } else if (avctx->color_primaries == AVCOL_PRI_BT470BG) {

        p_schro_params->format->colour_primaries = SCHRO_COLOUR_PRIMARY_SDTV_625;

    } else if (avctx->color_primaries == AVCOL_PRI_SMPTE170M) {

        p_schro_params->format->colour_primaries = SCHRO_COLOUR_PRIMARY_SDTV_525;

    }



    if (avctx->colorspace == AVCOL_SPC_BT709) {

        p_schro_params->format->colour_matrix = SCHRO_COLOUR_MATRIX_HDTV;

    } else if (avctx->colorspace == AVCOL_SPC_BT470BG) {

        p_schro_params->format->colour_matrix = SCHRO_COLOUR_MATRIX_SDTV;

    }



    if (avctx->color_trc == AVCOL_TRC_BT709) {

        p_schro_params->format->transfer_function = SCHRO_TRANSFER_CHAR_TV_GAMMA;

    }



    if (ff_get_schro_frame_format(p_schro_params->format->chroma_format,

                                  &p_schro_params->frame_format) == -1) {

        av_log(avctx, AV_LOG_ERROR,

               "This codec currently supports only planar YUV 4:2:0, 4:2:2"

               " and 4:4:4 formats.\n");

        return -1;

    }



    p_schro_params->format->frame_rate_numerator   = avctx->time_base.den;

    p_schro_params->format->frame_rate_denominator = avctx->time_base.num;



    p_schro_params->frame_size = av_image_get_buffer_size(avctx->pix_fmt,

                                                          avctx->width,

                                                          avctx->height, 1);



    if (!avctx->gop_size) {

        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "gop_structure",

                                         SCHRO_ENCODER_GOP_INTRA_ONLY);



#if FF_API_CODER_TYPE

FF_DISABLE_DEPRECATION_WARNINGS

        if (avctx->coder_type != FF_CODER_TYPE_VLC)

            p_schro_params->noarith = 0;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "enable_noarith",

                                         p_schro_params->noarith);

    } else {

        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "au_distance", avctx->gop_size);

        avctx->has_b_frames = 1;

        p_schro_params->dts = -1;

    }



    /* FIXME - Need to handle SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY. */

    if (avctx->flags & AV_CODEC_FLAG_QSCALE) {

        if (!avctx->global_quality) {

            /* lossless coding */

            schro_encoder_setting_set_double(p_schro_params->encoder,

                                             "rate_control",

                                             SCHRO_ENCODER_RATE_CONTROL_LOSSLESS);

        } else {

            int quality;

            schro_encoder_setting_set_double(p_schro_params->encoder,

                                             "rate_control",

                                             SCHRO_ENCODER_RATE_CONTROL_CONSTANT_QUALITY);



            quality = avctx->global_quality / FF_QP2LAMBDA;

            if (quality > 10)

                quality = 10;

            schro_encoder_setting_set_double(p_schro_params->encoder,

                                             "quality", quality);

        }

    } else {

        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "rate_control",

                                         SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE);



        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "bitrate", avctx->bit_rate);

    }



    if (avctx->flags & AV_CODEC_FLAG_INTERLACED_ME)

        /* All material can be coded as interlaced or progressive

           irrespective of the type of source material. */

        schro_encoder_setting_set_double(p_schro_params->encoder,

                                         "interlaced_coding", 1);



    schro_encoder_setting_set_double(p_schro_params->encoder, "open_gop",

                                     !(avctx->flags & AV_CODEC_FLAG_CLOSED_GOP));



    /* FIXME: Signal range hardcoded to 8-bit data until both libschroedinger

     * and libdirac support other bit-depth data. */

    schro_video_format_set_std_signal_range(p_schro_params->format,

                                            SCHRO_SIGNAL_RANGE_8BIT_VIDEO);



    /* Set the encoder format. */

    schro_encoder_set_video_format(p_schro_params->encoder,

                                   p_schro_params->format);



    /* Set the debug level. */

    schro_debug_set_level(avctx->debug);



    schro_encoder_start(p_schro_params->encoder);



    /* Initialize the encoded frame queue. */

    ff_schro_queue_init(&p_schro_params->enc_frame_queue);

    return 0;

}
