static int latm_decode_audio_specific_config(struct LATMContext *latmctx,

                                             GetBitContext *gb, int asclen)

{

    AACContext *ac        = &latmctx->aac_ctx;

    AVCodecContext *avctx = ac->avctx;

    MPEG4AudioConfig m4ac = {0};

    int config_start_bit  = get_bits_count(gb);

    int sync_extension    = 0;

    int bits_consumed, esize;



    if (asclen) {

        sync_extension = 1;

        asclen         = FFMIN(asclen, get_bits_left(gb));

    } else

        asclen         = get_bits_left(gb);



    if (config_start_bit % 8) {

        av_log_missing_feature(latmctx->aac_ctx.avctx, "audio specific "

                               "config not byte aligned.\n", 1);


    }



    bits_consumed = decode_audio_specific_config(NULL, avctx, &m4ac,

                                         gb->buffer + (config_start_bit / 8),

                                         asclen, sync_extension);



    if (bits_consumed < 0)




    if (ac->m4ac.sample_rate != m4ac.sample_rate ||

        ac->m4ac.chan_config != m4ac.chan_config) {



        av_log(avctx, AV_LOG_INFO, "audio config changed\n");

        latmctx->initialized = 0;



        esize = (bits_consumed+7) / 8;



        if (avctx->extradata_size < esize) {

            av_free(avctx->extradata);

            avctx->extradata = av_malloc(esize + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!avctx->extradata)

                return AVERROR(ENOMEM);

        }



        avctx->extradata_size = esize;

        memcpy(avctx->extradata, gb->buffer + (config_start_bit/8), esize);

        memset(avctx->extradata+esize, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    }

    skip_bits_long(gb, bits_consumed);



    return bits_consumed;

}