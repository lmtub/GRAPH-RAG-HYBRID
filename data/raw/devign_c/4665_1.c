static int libschroedinger_encode_init(AVCodecContext *avccontext)

{

    FfmpegSchroEncoderParams* p_schro_params = avccontext->priv_data;

    SchroVideoFormatEnum preset;



    /* Initialize the libraries that libschroedinger depends on. */

    schro_init();



    /* Create an encoder object. */

    p_schro_params->encoder = schro_encoder_new();



    if (!p_schro_params->encoder) {

        av_log(avccontext, AV_LOG_ERROR,

               "Unrecoverable Error: schro_encoder_new failed. ");

        return -1;

    }



    /* Initialize the format. */

    preset = ff_get_schro_video_format_preset(avccontext);

    p_schro_params->format =

                    schro_encoder_get_video_format(p_schro_params->encoder);

    schro_video_format_set_std_video_format (p_schro_params->format, preset);

    p_schro_params->format->width = avccontext->width;

    p_schro_params->format->height = avccontext->height;



    if (SetSchroChromaFormat(avccontext) == -1)

        return -1;



    if (ff_get_schro_frame_format(p_schro_params->format->chroma_format,

                                  &p_schro_params->frame_format) == -1) {

        av_log (avccontext, AV_LOG_ERROR,

                "This codec currently supports only planar YUV 4:2:0, 4:2:2"

                " and 4:4:4 formats.\n");

        return -1;

    }



    p_schro_params->format->frame_rate_numerator   = avccontext->time_base.den;

    p_schro_params->format->frame_rate_denominator = avccontext->time_base.num;



    p_schro_params->frame_size = avpicture_get_size(avccontext->pix_fmt,

                                                    avccontext->width,

                                                    avccontext->height);



    avccontext->coded_frame = &p_schro_params->picture;



    if (avccontext->gop_size == 0){

        schro_encoder_setting_set_double (p_schro_params->encoder,

                                          "gop_structure",

                                          SCHRO_ENCODER_GOP_INTRA_ONLY);



        if (avccontext->coder_type == FF_CODER_TYPE_VLC) {

            schro_encoder_setting_set_double (p_schro_params->encoder,

                                              "enable_noarith", 1);

        }

    }

    else {

        schro_encoder_setting_set_double (p_schro_params->encoder,

                                          "gop_structure",

                                          SCHRO_ENCODER_GOP_BIREF);

        avccontext->has_b_frames = 1;

    }



    /* FIXME - Need to handle SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY. */

    if (avccontext->flags & CODEC_FLAG_QSCALE) {

        if (avccontext->global_quality == 0) {

            /* lossless coding */

            schro_encoder_setting_set_double (p_schro_params->encoder,

                                          "rate_control",

                                          SCHRO_ENCODER_RATE_CONTROL_LOSSLESS);

        } else {

            int noise_threshold;

            schro_encoder_setting_set_double (p_schro_params->encoder,

                          "rate_control",

                          SCHRO_ENCODER_RATE_CONTROL_CONSTANT_NOISE_THRESHOLD);



            noise_threshold = avccontext->global_quality/FF_QP2LAMBDA;

            if (noise_threshold > 100)

                noise_threshold = 100;

            schro_encoder_setting_set_double (p_schro_params->encoder,

                                              "noise_threshold",

                                              noise_threshold);

        }

    }

    else {

        schro_encoder_setting_set_double ( p_schro_params->encoder,

                               "rate_control",

                               SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE);



        schro_encoder_setting_set_double (p_schro_params->encoder,

                                          "bitrate",

                                          avccontext->bit_rate);



    }



    if (avccontext->flags & CODEC_FLAG_INTERLACED_ME) {

        /* All material can be coded as interlaced or progressive

           irrespective of the type of source material. */

        schro_encoder_setting_set_double (p_schro_params->encoder,

                                            "interlaced_coding", 1);

    }



    /* FIXME: Signal range hardcoded to 8-bit data until both libschroedinger

     * and libdirac support other bit-depth data. */

    schro_video_format_set_std_signal_range(p_schro_params->format,

                                            SCHRO_SIGNAL_RANGE_8BIT_VIDEO);





    /* Hardcode motion vector precision to quarter pixel. */

    schro_encoder_setting_set_double (p_schro_params->encoder,

                                      "mv_precision", 2);



    /* Set the encoder format. */

    schro_encoder_set_video_format(p_schro_params->encoder,

                                   p_schro_params->format);



    /* Set the debug level. */

    schro_debug_set_level (avccontext->debug);



    schro_encoder_start (p_schro_params->encoder);



    /* Initialize the encoded frame queue. */

    ff_dirac_schro_queue_init (&p_schro_params->enc_frame_queue);

    return 0 ;

}
