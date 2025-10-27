static int alac_decode_frame(AVCodecContext *avctx,

                             void *outbuffer, int *outputsize,

                             AVPacket *avpkt)

{

    const uint8_t *inbuffer = avpkt->data;

    int input_buffer_size = avpkt->size;

    ALACContext *alac = avctx->priv_data;



    int channels;

    unsigned int outputsamples;

    int hassize;

    unsigned int readsamplesize;

    int isnotcompressed;

    uint8_t interlacing_shift;

    uint8_t interlacing_leftweight;



    /* short-circuit null buffers */

    if (!inbuffer || !input_buffer_size)

        return input_buffer_size;



    /* initialize from the extradata */

    if (!alac->context_initialized) {

        if (alac->avctx->extradata_size != ALAC_EXTRADATA_SIZE) {

            av_log(avctx, AV_LOG_ERROR, "alac: expected %d extradata bytes\n",

                ALAC_EXTRADATA_SIZE);

            return input_buffer_size;

        }

        if (alac_set_info(alac)) {

            av_log(avctx, AV_LOG_ERROR, "alac: set_info failed\n");

            return input_buffer_size;

        }

        alac->context_initialized = 1;

    }



    init_get_bits(&alac->gb, inbuffer, input_buffer_size * 8);



    channels = get_bits(&alac->gb, 3) + 1;

    if (channels > MAX_CHANNELS) {

        av_log(avctx, AV_LOG_ERROR, "channels > %d not supported\n",

               MAX_CHANNELS);

        return input_buffer_size;

    }



    /* 2^result = something to do with output waiting.

     * perhaps matters if we read > 1 frame in a pass?

     */

    skip_bits(&alac->gb, 4);



    skip_bits(&alac->gb, 12); /* unknown, skip 12 bits */



    /* the output sample size is stored soon */

    hassize = get_bits1(&alac->gb);



    alac->wasted_bits = get_bits(&alac->gb, 2) << 3;



    /* whether the frame is compressed */

    isnotcompressed = get_bits1(&alac->gb);



    if (hassize) {

        /* now read the number of samples as a 32bit integer */

        outputsamples = get_bits_long(&alac->gb, 32);

        if(outputsamples > alac->setinfo_max_samples_per_frame){

            av_log(avctx, AV_LOG_ERROR, "outputsamples %d > %d\n", outputsamples, alac->setinfo_max_samples_per_frame);

            return -1;

        }

    } else

        outputsamples = alac->setinfo_max_samples_per_frame;



    switch (alac->setinfo_sample_size) {

    case 16: avctx->sample_fmt    = SAMPLE_FMT_S16;

             alac->bytespersample = channels << 1;

             break;

    case 24: avctx->sample_fmt    = SAMPLE_FMT_S32;

             alac->bytespersample = channels << 2;

             break;

    default: av_log(avctx, AV_LOG_ERROR, "Sample depth %d is not supported.\n",

                    alac->setinfo_sample_size);

             return -1;

    }



    if(outputsamples > *outputsize / alac->bytespersample){

        av_log(avctx, AV_LOG_ERROR, "sample buffer too small\n");

        return -1;

    }



    *outputsize = outputsamples * alac->bytespersample;

    readsamplesize = alac->setinfo_sample_size - (alac->wasted_bits) + channels - 1;

    if (readsamplesize > MIN_CACHE_BITS) {

        av_log(avctx, AV_LOG_ERROR, "readsamplesize too big (%d)\n", readsamplesize);

        return -1;

    }



    if (!isnotcompressed) {

        /* so it is compressed */

        int16_t predictor_coef_table[MAX_CHANNELS][32];

        int predictor_coef_num[MAX_CHANNELS];

        int prediction_type[MAX_CHANNELS];

        int prediction_quantitization[MAX_CHANNELS];

        int ricemodifier[MAX_CHANNELS];

        int i, chan;



        interlacing_shift = get_bits(&alac->gb, 8);

        interlacing_leftweight = get_bits(&alac->gb, 8);



        for (chan = 0; chan < channels; chan++) {

            prediction_type[chan] = get_bits(&alac->gb, 4);

            prediction_quantitization[chan] = get_bits(&alac->gb, 4);



            ricemodifier[chan] = get_bits(&alac->gb, 3);

            predictor_coef_num[chan] = get_bits(&alac->gb, 5);



            /* read the predictor table */

            for (i = 0; i < predictor_coef_num[chan]; i++)

                predictor_coef_table[chan][i] = (int16_t)get_bits(&alac->gb, 16);

        }



        if (alac->wasted_bits) {

            int i, ch;

            for (i = 0; i < outputsamples; i++) {

                for (ch = 0; ch < channels; ch++)

                    alac->wasted_bits_buffer[ch][i] = get_bits(&alac->gb, alac->wasted_bits);

            }

        }

        for (chan = 0; chan < channels; chan++) {

            bastardized_rice_decompress(alac,

                                        alac->predicterror_buffer[chan],

                                        outputsamples,

                                        readsamplesize,

                                        alac->setinfo_rice_initialhistory,

                                        alac->setinfo_rice_kmodifier,

                                        ricemodifier[chan] * alac->setinfo_rice_historymult / 4,

                                        (1 << alac->setinfo_rice_kmodifier) - 1);



            if (prediction_type[chan] == 0) {

                /* adaptive fir */

                predictor_decompress_fir_adapt(alac->predicterror_buffer[chan],

                                               alac->outputsamples_buffer[chan],

                                               outputsamples,

                                               readsamplesize,

                                               predictor_coef_table[chan],

                                               predictor_coef_num[chan],

                                               prediction_quantitization[chan]);

            } else {

                av_log(avctx, AV_LOG_ERROR, "FIXME: unhandled prediction type: %i\n", prediction_type[chan]);

                /* I think the only other prediction type (or perhaps this is

                 * just a boolean?) runs adaptive fir twice.. like:

                 * predictor_decompress_fir_adapt(predictor_error, tempout, ...)

                 * predictor_decompress_fir_adapt(predictor_error, outputsamples ...)

                 * little strange..

                 */

            }

        }

    } else {

        /* not compressed, easy case */

        int i, chan;

        if (alac->setinfo_sample_size <= 16) {

        for (i = 0; i < outputsamples; i++)

            for (chan = 0; chan < channels; chan++) {

                int32_t audiobits;



                audiobits = get_sbits_long(&alac->gb, alac->setinfo_sample_size);



                alac->outputsamples_buffer[chan][i] = audiobits;

            }

        } else {

            for (i = 0; i < outputsamples; i++) {

                for (chan = 0; chan < channels; chan++) {

                    alac->outputsamples_buffer[chan][i] = get_bits(&alac->gb,

                                                          alac->setinfo_sample_size);

                    alac->outputsamples_buffer[chan][i] = sign_extend(alac->outputsamples_buffer[chan][i],

                                                                      alac->setinfo_sample_size);

                }

            }

        }

        alac->wasted_bits = 0;

        interlacing_shift = 0;

        interlacing_leftweight = 0;

    }

    if (get_bits(&alac->gb, 3) != 7)

        av_log(avctx, AV_LOG_ERROR, "Error : Wrong End Of Frame\n");



    switch(alac->setinfo_sample_size) {

    case 16:

        if (channels == 2) {

            reconstruct_stereo_16(alac->outputsamples_buffer,

                                  (int16_t*)outbuffer,

                                  alac->numchannels,

                                  outputsamples,

                                  interlacing_shift,

                                  interlacing_leftweight);

        } else {

            int i;

            for (i = 0; i < outputsamples; i++) {

                ((int16_t*)outbuffer)[i] = alac->outputsamples_buffer[0][i];

            }

        }

        break;

    case 24:

        if (channels == 2) {

            decorrelate_stereo_24(alac->outputsamples_buffer,

                                  outbuffer,

                                  alac->wasted_bits_buffer,

                                  alac->wasted_bits,

                                  alac->numchannels,

                                  outputsamples,

                                  interlacing_shift,

                                  interlacing_leftweight);

        } else {

            int i;

            for (i = 0; i < outputsamples; i++)

                ((int32_t *)outbuffer)[i] = alac->outputsamples_buffer[0][i] << 8;

        }

        break;

    }



    if (input_buffer_size * 8 - get_bits_count(&alac->gb) > 8)

        av_log(avctx, AV_LOG_ERROR, "Error : %d bits left\n", input_buffer_size * 8 - get_bits_count(&alac->gb));



    return input_buffer_size;

}
