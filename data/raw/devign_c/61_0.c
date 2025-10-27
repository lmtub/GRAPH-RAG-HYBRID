static int truespeech_decode_frame(AVCodecContext *avctx,

                void *data, int *data_size,

                AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TSContext *c = avctx->priv_data;



    int i, j;

    short *samples = data;

    int consumed = 0;

    int16_t out_buf[240];

    int iterations;



    if (!buf_size)

        return 0;



    if (buf_size < 32) {

        av_log(avctx, AV_LOG_ERROR,

               "Too small input buffer (%d bytes), need at least 32 bytes\n", buf_size);

        return -1;

    }

    iterations = FFMIN(buf_size / 32, *data_size / 480);

    for(j = 0; j < iterations; j++) {

        truespeech_read_frame(c, buf + consumed);

        consumed += 32;



        truespeech_correlate_filter(c);

        truespeech_filters_merge(c);



        memset(out_buf, 0, 240 * 2);

        for(i = 0; i < 4; i++) {

            truespeech_apply_twopoint_filter(c, i);

            truespeech_place_pulses(c, out_buf + i * 60, i);

            truespeech_update_filters(c, out_buf + i * 60, i);

            truespeech_synth(c, out_buf + i * 60, i);

        }



        truespeech_save_prevvec(c);



        /* finally output decoded frame */

        for(i = 0; i < 240; i++)

            *samples++ = out_buf[i];



    }



    *data_size = consumed * 15;



    return consumed;

}
