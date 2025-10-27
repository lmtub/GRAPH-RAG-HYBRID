static int encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                        const AVFrame *frame, int *got_packet_ptr)

{

    int i, k, channel;

    DCAContext *c = avctx->priv_data;

    const int16_t *samples;

    int ret, real_channel = 0;



    if ((ret = ff_alloc_packet2(avctx, avpkt, DCA_MAX_FRAME_SIZE + DCA_HEADER_SIZE)))

        return ret;



    samples = (const int16_t *)frame->data[0];

    for (i = 0; i < PCM_SAMPLES; i ++) { /* i is the decimated sample number */

        for (channel = 0; channel < c->prim_channels + 1; channel++) {

            real_channel = c->channel_order_tab[channel];

            if (real_channel >= 0) {

                /* Get 32 PCM samples */

                for (k = 0; k < 32; k++) { /* k is the sample number in a 32-sample block */

                    c->pcm[k] = samples[avctx->channels * (32 * i + k) + channel] << 16;

                }

                /* Put subband samples into the proper place */

                qmf_decompose(c, c->pcm, &c->subband[i][real_channel][0], real_channel);

            }

        }

    }



    if (c->lfe_channel) {

        for (i = 0; i < PCM_SAMPLES / 2; i++) {

            for (k = 0; k < LFE_INTERPOLATION; k++) /* k is the sample number in a 32-sample block */

                c->pcm[k] = samples[avctx->channels * (LFE_INTERPOLATION*i+k) + c->lfe_offset] << 16;

            c->lfe_data[i] = lfe_downsample(c, c->pcm);

        }

    }



    put_frame(c, c->subband, avpkt->data);



    avpkt->size     = c->frame_size;

    *got_packet_ptr = 1;

    return 0;

}
