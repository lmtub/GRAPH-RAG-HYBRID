static av_cold int flac_encode_init(AVCodecContext *avctx)

{

    int freq = avctx->sample_rate;

    int channels = avctx->channels;

    FlacEncodeContext *s = avctx->priv_data;

    int i, level;

    uint8_t *streaminfo;



    s->avctx = avctx;



    dsputil_init(&s->dsp, avctx);



    if (avctx->sample_fmt != SAMPLE_FMT_S16)

        return -1;



    if (channels < 1 || channels > FLAC_MAX_CHANNELS)

        return -1;

    s->channels = channels;



    /* find samplerate in table */

    if (freq < 1)

        return -1;

    for (i = 4; i < 12; i++) {

        if (freq == ff_flac_sample_rate_table[i]) {

            s->samplerate = ff_flac_sample_rate_table[i];

            s->sr_code[0] = i;

            s->sr_code[1] = 0;

            break;

        }

    }

    /* if not in table, samplerate is non-standard */

    if (i == 12) {

        if (freq % 1000 == 0 && freq < 255000) {

            s->sr_code[0] = 12;

            s->sr_code[1] = freq / 1000;

        } else if (freq % 10 == 0 && freq < 655350) {

            s->sr_code[0] = 14;

            s->sr_code[1] = freq / 10;

        } else if (freq < 65535) {

            s->sr_code[0] = 13;

            s->sr_code[1] = freq;

        } else {

            return -1;

        }

        s->samplerate = freq;

    }



    /* set compression option defaults based on avctx->compression_level */

    if (avctx->compression_level < 0)

        s->options.compression_level = 5;

    else

        s->options.compression_level = avctx->compression_level;

    av_log(avctx, AV_LOG_DEBUG, " compression: %d\n", s->options.compression_level);



    level = s->options.compression_level;

    if (level > 12) {

        av_log(avctx, AV_LOG_ERROR, "invalid compression level: %d\n",

               s->options.compression_level);

        return -1;

    }



    s->options.block_time_ms = ((int[]){ 27, 27, 27,105,105,105,105,105,105,105,105,105,105})[level];



    s->options.lpc_type      = ((int[]){ AV_LPC_TYPE_FIXED,    AV_LPC_TYPE_FIXED,    AV_LPC_TYPE_FIXED,

                                         AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON,

                                         AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON,

                                         AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON, AV_LPC_TYPE_LEVINSON,

                                         AV_LPC_TYPE_LEVINSON})[level];



    s->options.min_prediction_order = ((int[]){  2,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1})[level];

    s->options.max_prediction_order = ((int[]){  3,  4,  4,  6,  8,  8,  8,  8, 12, 12, 12, 32, 32})[level];



    s->options.prediction_order_method = ((int[]){ ORDER_METHOD_EST,    ORDER_METHOD_EST,    ORDER_METHOD_EST,

                                                   ORDER_METHOD_EST,    ORDER_METHOD_EST,    ORDER_METHOD_EST,

                                                   ORDER_METHOD_4LEVEL, ORDER_METHOD_LOG,    ORDER_METHOD_4LEVEL,

                                                   ORDER_METHOD_LOG,    ORDER_METHOD_SEARCH, ORDER_METHOD_LOG,

                                                   ORDER_METHOD_SEARCH})[level];



    s->options.min_partition_order = ((int[]){  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0})[level];

    s->options.max_partition_order = ((int[]){  2,  2,  3,  3,  3,  8,  8,  8,  8,  8,  8,  8,  8})[level];



    /* set compression option overrides from AVCodecContext */

#if LIBAVCODEC_VERSION_MAJOR < 53

    /* for compatibility with deprecated AVCodecContext.use_lpc */

    if (avctx->use_lpc == 0) {

        s->options.lpc_type = AV_LPC_TYPE_FIXED;

    } else if (avctx->use_lpc == 1) {

        s->options.lpc_type = AV_LPC_TYPE_LEVINSON;

    } else if (avctx->use_lpc > 1) {

        s->options.lpc_type   = AV_LPC_TYPE_CHOLESKY;

        s->options.lpc_passes = avctx->use_lpc - 1;

    }

#endif

    if (avctx->lpc_type > AV_LPC_TYPE_DEFAULT) {

        if (avctx->lpc_type > AV_LPC_TYPE_CHOLESKY) {

            av_log(avctx, AV_LOG_ERROR, "unknown lpc type: %d\n", avctx->lpc_type);

            return -1;

        }

        s->options.lpc_type = avctx->lpc_type;

        if (s->options.lpc_type == AV_LPC_TYPE_CHOLESKY) {

            if (avctx->lpc_passes < 0) {

                // default number of passes for Cholesky

                s->options.lpc_passes = 2;

            } else if (avctx->lpc_passes == 0) {

                av_log(avctx, AV_LOG_ERROR, "invalid number of lpc passes: %d\n",

                       avctx->lpc_passes);

                return -1;

            } else {

                s->options.lpc_passes = avctx->lpc_passes;

            }

        }

    }

    switch (s->options.lpc_type) {

    case AV_LPC_TYPE_NONE:

        av_log(avctx, AV_LOG_DEBUG, " lpc type: None\n");

        break;

    case AV_LPC_TYPE_FIXED:

        av_log(avctx, AV_LOG_DEBUG, " lpc type: Fixed pre-defined coefficients\n");

        break;

    case AV_LPC_TYPE_LEVINSON:

        av_log(avctx, AV_LOG_DEBUG, " lpc type: Levinson-Durbin recursion with Welch window\n");

        break;

    case AV_LPC_TYPE_CHOLESKY:

        av_log(avctx, AV_LOG_DEBUG, " lpc type: Cholesky factorization, %d pass%s\n",

               s->options.lpc_passes, s->options.lpc_passes==1?"":"es");

        break;

    }



    if (s->options.lpc_type == AV_LPC_TYPE_NONE) {

        s->options.min_prediction_order = 0;

    } else if (avctx->min_prediction_order >= 0) {

        if (s->options.lpc_type == AV_LPC_TYPE_FIXED) {

            if (avctx->min_prediction_order > MAX_FIXED_ORDER) {

                av_log(avctx, AV_LOG_ERROR, "invalid min prediction order: %d\n",

                       avctx->min_prediction_order);

                return -1;

            }

        } else if (avctx->min_prediction_order < MIN_LPC_ORDER ||

                   avctx->min_prediction_order > MAX_LPC_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid min prediction order: %d\n",

                   avctx->min_prediction_order);

            return -1;

        }

        s->options.min_prediction_order = avctx->min_prediction_order;

    }

    if (s->options.lpc_type == AV_LPC_TYPE_NONE) {

        s->options.max_prediction_order = 0;

    } else if (avctx->max_prediction_order >= 0) {

        if (s->options.lpc_type == AV_LPC_TYPE_FIXED) {

            if (avctx->max_prediction_order > MAX_FIXED_ORDER) {

                av_log(avctx, AV_LOG_ERROR, "invalid max prediction order: %d\n",

                       avctx->max_prediction_order);

                return -1;

            }

        } else if (avctx->max_prediction_order < MIN_LPC_ORDER ||

                   avctx->max_prediction_order > MAX_LPC_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid max prediction order: %d\n",

                   avctx->max_prediction_order);

            return -1;

        }

        s->options.max_prediction_order = avctx->max_prediction_order;

    }

    if (s->options.max_prediction_order < s->options.min_prediction_order) {

        av_log(avctx, AV_LOG_ERROR, "invalid prediction orders: min=%d max=%d\n",

               s->options.min_prediction_order, s->options.max_prediction_order);

        return -1;

    }

    av_log(avctx, AV_LOG_DEBUG, " prediction order: %d, %d\n",

           s->options.min_prediction_order, s->options.max_prediction_order);



    if (avctx->prediction_order_method >= 0) {

        if (avctx->prediction_order_method > ORDER_METHOD_LOG) {

            av_log(avctx, AV_LOG_ERROR, "invalid prediction order method: %d\n",

                   avctx->prediction_order_method);

            return -1;

        }

        s->options.prediction_order_method = avctx->prediction_order_method;

    }

    switch (s->options.prediction_order_method) {

        case ORDER_METHOD_EST:    av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "estimate"); break;

        case ORDER_METHOD_2LEVEL: av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "2-level"); break;

        case ORDER_METHOD_4LEVEL: av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "4-level"); break;

        case ORDER_METHOD_8LEVEL: av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "8-level"); break;

        case ORDER_METHOD_SEARCH: av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "full search"); break;

        case ORDER_METHOD_LOG:    av_log(avctx, AV_LOG_DEBUG, " order method: %s\n",

                                         "log search"); break;

    }



    if (avctx->min_partition_order >= 0) {

        if (avctx->min_partition_order > MAX_PARTITION_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid min partition order: %d\n",

                   avctx->min_partition_order);

            return -1;

        }

        s->options.min_partition_order = avctx->min_partition_order;

    }

    if (avctx->max_partition_order >= 0) {

        if (avctx->max_partition_order > MAX_PARTITION_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid max partition order: %d\n",

                   avctx->max_partition_order);

            return -1;

        }

        s->options.max_partition_order = avctx->max_partition_order;

    }

    if (s->options.max_partition_order < s->options.min_partition_order) {

        av_log(avctx, AV_LOG_ERROR, "invalid partition orders: min=%d max=%d\n",

               s->options.min_partition_order, s->options.max_partition_order);

        return -1;

    }

    av_log(avctx, AV_LOG_DEBUG, " partition order: %d, %d\n",

           s->options.min_partition_order, s->options.max_partition_order);



    if (avctx->frame_size > 0) {

        if (avctx->frame_size < FLAC_MIN_BLOCKSIZE ||

                avctx->frame_size > FLAC_MAX_BLOCKSIZE) {

            av_log(avctx, AV_LOG_ERROR, "invalid block size: %d\n",

                   avctx->frame_size);

            return -1;

        }

    } else {

        s->avctx->frame_size = select_blocksize(s->samplerate, s->options.block_time_ms);

    }

    s->max_blocksize = s->avctx->frame_size;

    av_log(avctx, AV_LOG_DEBUG, " block size: %d\n", s->avctx->frame_size);



    /* set LPC precision */

    if (avctx->lpc_coeff_precision > 0) {

        if (avctx->lpc_coeff_precision > MAX_LPC_PRECISION) {

            av_log(avctx, AV_LOG_ERROR, "invalid lpc coeff precision: %d\n",

                   avctx->lpc_coeff_precision);

            return -1;

        }

        s->options.lpc_coeff_precision = avctx->lpc_coeff_precision;

    } else {

        /* default LPC precision */

        s->options.lpc_coeff_precision = 15;

    }

    av_log(avctx, AV_LOG_DEBUG, " lpc precision: %d\n",

           s->options.lpc_coeff_precision);



    /* set maximum encoded frame size in verbatim mode */

    s->max_framesize = ff_flac_get_max_frame_size(s->avctx->frame_size,

                                                  s->channels, 16);



    /* initialize MD5 context */

    s->md5ctx = av_malloc(av_md5_size);

    if (!s->md5ctx)


    av_md5_init(s->md5ctx);



    streaminfo = av_malloc(FLAC_STREAMINFO_SIZE);

    if (!streaminfo)


    write_streaminfo(s, streaminfo);

    avctx->extradata = streaminfo;

    avctx->extradata_size = FLAC_STREAMINFO_SIZE;



    s->frame_count   = 0;

    s->min_framesize = s->max_framesize;



    avctx->coded_frame = avcodec_alloc_frame();



    avctx->coded_frame->key_frame = 1;



    return 0;

}