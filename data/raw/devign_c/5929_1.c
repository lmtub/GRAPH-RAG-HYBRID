static int cook_decode_init(AVCodecContext *avctx)
{
    COOKextradata *e = avctx->extradata;
    COOKContext *q = avctx->priv_data;
    /* Take care of the codec specific extradata. */
    if (avctx->extradata_size <= 0) {
        av_log(NULL,AV_LOG_ERROR,"Necessary extradata missing!\n");
    } else {
        /* 8 for mono, 16 for stereo, ? for multichannel
           Swap to right endianness so we don't need to care later on. */
        av_log(NULL,AV_LOG_DEBUG,"codecdata_length=%d\n",avctx->extradata_size);
        if (avctx->extradata_size >= 8){
            e->cookversion = be2me_32(e->cookversion);
            e->samples_per_frame = be2me_16(e->samples_per_frame);
            e->subbands = be2me_16(e->subbands);
        }
        if (avctx->extradata_size >= 16){
            e->js_subband_start = be2me_16(e->js_subband_start);
            e->js_vlc_bits = be2me_16(e->js_vlc_bits);
        }
    }
    /* Take data from the AVCodecContext (RM container). */
    q->sample_rate = avctx->sample_rate;
    q->nb_channels = avctx->channels;
    q->bit_rate = avctx->bit_rate;
    /* Initialize state. */
    q->random_state = 1;
    /* Initialize extradata related variables. */
    q->samples_per_channel = e->samples_per_frame / q->nb_channels;
    q->samples_per_frame = e->samples_per_frame;
    q->subbands = e->subbands;
    q->bits_per_subpacket = avctx->block_align * 8;
    /* Initialize default data states. */
    q->js_subband_start = 0;
    q->log2_numvector_size = 5;
    q->total_subbands = q->subbands;
    /* Initialize version-dependent variables */
    av_log(NULL,AV_LOG_DEBUG,"e->cookversion=%x\n",e->cookversion);
    switch (e->cookversion) {
        case MONO_COOK1:
            if (q->nb_channels != 1) {
                av_log(NULL,AV_LOG_ERROR,"Container channels != 1, report sample!\n");
            }
            av_log(NULL,AV_LOG_DEBUG,"MONO_COOK1\n");
            break;
        case MONO_COOK2:
            if (q->nb_channels != 1) {
                q->joint_stereo = 0;
                q->bits_per_subpacket = q->bits_per_subpacket/2;
            }
            av_log(NULL,AV_LOG_DEBUG,"MONO_COOK2\n");
            break;
        case JOINT_STEREO:
            if (q->nb_channels != 2) {
                av_log(NULL,AV_LOG_ERROR,"Container channels != 2, report sample!\n");
            }
            av_log(NULL,AV_LOG_DEBUG,"JOINT_STEREO\n");
            if (avctx->extradata_size >= 16){
                q->total_subbands = q->subbands + e->js_subband_start;
                q->js_subband_start = e->js_subband_start;
                q->joint_stereo = 1;
                q->js_vlc_bits = e->js_vlc_bits;
            }
            if (q->samples_per_channel > 256) {
                q->log2_numvector_size  = 6;
            }
            if (q->samples_per_channel > 512) {
                q->log2_numvector_size  = 7;
            }
            break;
        case MC_COOK:
            av_log(NULL,AV_LOG_ERROR,"MC_COOK not supported!\n");
            break;
        default:
            av_log(NULL,AV_LOG_ERROR,"Unknown Cook version, report sample!\n");
            break;
    }
    /* Initialize variable relations */
    q->mlt_size = q->samples_per_channel;
    q->numvector_size = (1 << q->log2_numvector_size);
    /* Generate tables */
    init_rootpow2table(q);
    init_pow2table(q);
    init_gain_table(q);
    if (init_cook_vlc_tables(q) != 0)
    /* Pad the databuffer with FF_INPUT_BUFFER_PADDING_SIZE,
       this is for the bitstreamreader. */
    if ((q->decoded_bytes_buffer = av_mallocz((avctx->block_align+(4-avctx->block_align%4) + FF_INPUT_BUFFER_PADDING_SIZE)*sizeof(uint8_t)))  == NULL)
    q->decode_buf_ptr[0] = q->decode_buffer_1;
    q->decode_buf_ptr[1] = q->decode_buffer_2;
    q->decode_buf_ptr[2] = q->decode_buffer_3;
    q->decode_buf_ptr[3] = q->decode_buffer_4;
    q->decode_buf_ptr2[0] = q->decode_buffer_3;
    q->decode_buf_ptr2[1] = q->decode_buffer_4;
    q->previous_buffer_ptr[0] = q->mono_previous_buffer1;
    q->previous_buffer_ptr[1] = q->mono_previous_buffer2;
    /* Initialize transform. */
    if ( init_cook_mlt(q) == 0 )
    /* Try to catch some obviously faulty streams, othervise it might be exploitable */
    if (q->total_subbands > 53) {
        av_log(NULL,AV_LOG_ERROR,"total_subbands > 53, report sample!\n");
    }
    if (q->subbands > 50) {
        av_log(NULL,AV_LOG_ERROR,"subbands > 50, report sample!\n");
    }
    if ((q->samples_per_channel == 256) || (q->samples_per_channel == 512) || (q->samples_per_channel == 1024)) {
    } else {
        av_log(NULL,AV_LOG_ERROR,"unknown amount of samples_per_channel = %d, report sample!\n",q->samples_per_channel);
    }
#ifdef COOKDEBUG
    dump_cook_context(q,e);
#endif
    return 0;
}