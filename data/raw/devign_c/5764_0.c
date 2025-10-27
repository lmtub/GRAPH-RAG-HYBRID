static av_cold int cook_decode_init(AVCodecContext *avctx)

{

    COOKContext *q = avctx->priv_data;

    const uint8_t *edata_ptr = avctx->extradata;

    const uint8_t *edata_ptr_end = edata_ptr + avctx->extradata_size;

    int extradata_size = avctx->extradata_size;

    int s = 0;

    unsigned int channel_mask = 0;

    q->avctx = avctx;



    /* Take care of the codec specific extradata. */

    if (extradata_size <= 0) {

        av_log(avctx,AV_LOG_ERROR,"Necessary extradata missing!\n");

        return -1;

    }

    av_log(avctx,AV_LOG_DEBUG,"codecdata_length=%d\n",avctx->extradata_size);



    /* Take data from the AVCodecContext (RM container). */

    q->sample_rate = avctx->sample_rate;

    q->nb_channels = avctx->channels;

    q->bit_rate = avctx->bit_rate;



    /* Initialize RNG. */

    av_lfg_init(&q->random_state, 0);



    while(edata_ptr < edata_ptr_end){

        /* 8 for mono, 16 for stereo, ? for multichannel

           Swap to right endianness so we don't need to care later on. */

        if (extradata_size >= 8){

            q->subpacket[s].cookversion = bytestream_get_be32(&edata_ptr);

            q->subpacket[s].samples_per_frame =  bytestream_get_be16(&edata_ptr);

            q->subpacket[s].subbands = bytestream_get_be16(&edata_ptr);

            extradata_size -= 8;

        }

        if (avctx->extradata_size >= 8){

            bytestream_get_be32(&edata_ptr);    //Unknown unused

            q->subpacket[s].js_subband_start = bytestream_get_be16(&edata_ptr);

            q->subpacket[s].js_vlc_bits = bytestream_get_be16(&edata_ptr);

            extradata_size -= 8;

        }



        /* Initialize extradata related variables. */

        q->subpacket[s].samples_per_channel = q->subpacket[s].samples_per_frame / q->nb_channels;

        q->subpacket[s].bits_per_subpacket = avctx->block_align * 8;



        /* Initialize default data states. */

        q->subpacket[s].log2_numvector_size = 5;

        q->subpacket[s].total_subbands = q->subpacket[s].subbands;

        q->subpacket[s].num_channels = 1;



        /* Initialize version-dependent variables */



        av_log(avctx,AV_LOG_DEBUG,"subpacket[%i].cookversion=%x\n",s,q->subpacket[s].cookversion);

        q->subpacket[s].joint_stereo = 0;

        switch (q->subpacket[s].cookversion) {

            case MONO:

                if (q->nb_channels != 1) {

                    av_log_ask_for_sample(avctx, "Container channels != 1.\n");

                    return -1;

                }

                av_log(avctx,AV_LOG_DEBUG,"MONO\n");

                break;

            case STEREO:

                if (q->nb_channels != 1) {

                    q->subpacket[s].bits_per_subpdiv = 1;

                    q->subpacket[s].num_channels = 2;

                }

                av_log(avctx,AV_LOG_DEBUG,"STEREO\n");

                break;

            case JOINT_STEREO:

                if (q->nb_channels != 2) {

                    av_log_ask_for_sample(avctx, "Container channels != 2.\n");

                    return -1;

                }

                av_log(avctx,AV_LOG_DEBUG,"JOINT_STEREO\n");

                if (avctx->extradata_size >= 16){

                    q->subpacket[s].total_subbands = q->subpacket[s].subbands + q->subpacket[s].js_subband_start;

                    q->subpacket[s].joint_stereo = 1;

                    q->subpacket[s].num_channels = 2;

                }

                if (q->subpacket[s].samples_per_channel > 256) {

                    q->subpacket[s].log2_numvector_size  = 6;

                }

                if (q->subpacket[s].samples_per_channel > 512) {

                    q->subpacket[s].log2_numvector_size  = 7;

                }

                break;

            case MC_COOK:

                av_log(avctx,AV_LOG_DEBUG,"MULTI_CHANNEL\n");

                if(extradata_size >= 4)

                    channel_mask |= q->subpacket[s].channel_mask = bytestream_get_be32(&edata_ptr);



                if(cook_count_channels(q->subpacket[s].channel_mask) > 1){

                    q->subpacket[s].total_subbands = q->subpacket[s].subbands + q->subpacket[s].js_subband_start;

                    q->subpacket[s].joint_stereo = 1;

                    q->subpacket[s].num_channels = 2;

                    q->subpacket[s].samples_per_channel = q->subpacket[s].samples_per_frame >> 1;



                    if (q->subpacket[s].samples_per_channel > 256) {

                        q->subpacket[s].log2_numvector_size  = 6;

                    }

                    if (q->subpacket[s].samples_per_channel > 512) {

                        q->subpacket[s].log2_numvector_size  = 7;

                    }

                }else

                    q->subpacket[s].samples_per_channel = q->subpacket[s].samples_per_frame;



                break;

            default:

                av_log_ask_for_sample(avctx, "Unknown Cook version.\n");

                return -1;

        }



        if(s > 1 && q->subpacket[s].samples_per_channel != q->samples_per_channel) {

            av_log(avctx,AV_LOG_ERROR,"different number of samples per channel!\n");

            return -1;

        } else

            q->samples_per_channel = q->subpacket[0].samples_per_channel;





        /* Initialize variable relations */

        q->subpacket[s].numvector_size = (1 << q->subpacket[s].log2_numvector_size);



        /* Try to catch some obviously faulty streams, othervise it might be exploitable */

        if (q->subpacket[s].total_subbands > 53) {

            av_log_ask_for_sample(avctx, "total_subbands > 53\n");

            return -1;

        }



        if ((q->subpacket[s].js_vlc_bits > 6) || (q->subpacket[s].js_vlc_bits < 0)) {

            av_log(avctx,AV_LOG_ERROR,"js_vlc_bits = %d, only >= 0 and <= 6 allowed!\n",q->subpacket[s].js_vlc_bits);

            return -1;

        }



        if (q->subpacket[s].subbands > 50) {

            av_log_ask_for_sample(avctx, "subbands > 50\n");

            return -1;

        }

        q->subpacket[s].gains1.now      = q->subpacket[s].gain_1;

        q->subpacket[s].gains1.previous = q->subpacket[s].gain_2;

        q->subpacket[s].gains2.now      = q->subpacket[s].gain_3;

        q->subpacket[s].gains2.previous = q->subpacket[s].gain_4;



        q->num_subpackets++;

        s++;

        if (s > MAX_SUBPACKETS) {

            av_log_ask_for_sample(avctx, "Too many subpackets > 5\n");

            return -1;

        }

    }

    /* Generate tables */

    init_pow2table();

    init_gain_table(q);

    init_cplscales_table(q);



    if (init_cook_vlc_tables(q) != 0)

        return -1;





    if(avctx->block_align >= UINT_MAX/2)

        return -1;



    /* Pad the databuffer with:

       DECODE_BYTES_PAD1 or DECODE_BYTES_PAD2 for decode_bytes(),

       FF_INPUT_BUFFER_PADDING_SIZE, for the bitstreamreader. */

        q->decoded_bytes_buffer =

          av_mallocz(avctx->block_align

                     + DECODE_BYTES_PAD1(avctx->block_align)

                     + FF_INPUT_BUFFER_PADDING_SIZE);

    if (q->decoded_bytes_buffer == NULL)

        return -1;



    /* Initialize transform. */

    if ( init_cook_mlt(q) != 0 )

        return -1;



    /* Initialize COOK signal arithmetic handling */

    if (1) {

        q->scalar_dequant  = scalar_dequant_float;

        q->decouple        = decouple_float;

        q->imlt_window     = imlt_window_float;

        q->interpolate     = interpolate_float;

        q->saturate_output = saturate_output_float;

    }



    /* Try to catch some obviously faulty streams, othervise it might be exploitable */

    if ((q->samples_per_channel == 256) || (q->samples_per_channel == 512) || (q->samples_per_channel == 1024)) {

    } else {

        av_log_ask_for_sample(avctx,

                              "unknown amount of samples_per_channel = %d\n",

                              q->samples_per_channel);

        return -1;

    }



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    if (channel_mask)

        avctx->channel_layout = channel_mask;

    else

        avctx->channel_layout = (avctx->channels==2) ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;



#ifdef DEBUG

    dump_cook_context(q);

#endif

    return 0;

}
