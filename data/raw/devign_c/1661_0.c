static int g722_decode_frame(AVCodecContext *avctx, void *data,

                             int *data_size, AVPacket *avpkt)

{

    G722Context *c = avctx->priv_data;

    int16_t *out_buf = data;

    int j, out_len = 0;

    const int skip = 8 - avctx->bits_per_coded_sample;

    const int16_t *quantizer_table = low_inv_quants[skip];

    GetBitContext gb;



    init_get_bits(&gb, avpkt->data, avpkt->size * 8);



    for (j = 0; j < avpkt->size; j++) {

        int ilow, ihigh, rlow, rhigh, dhigh;

        int xout1, xout2;



        ihigh = get_bits(&gb, 2);

        ilow = get_bits(&gb, 6 - skip);

        skip_bits(&gb, skip);



        rlow = av_clip((c->band[0].scale_factor * quantizer_table[ilow] >> 10)

                      + c->band[0].s_predictor, -16384, 16383);



        ff_g722_update_low_predictor(&c->band[0], ilow >> (2 - skip));



        dhigh = c->band[1].scale_factor * ff_g722_high_inv_quant[ihigh] >> 10;

        rhigh = av_clip(dhigh + c->band[1].s_predictor, -16384, 16383);



        ff_g722_update_high_predictor(&c->band[1], dhigh, ihigh);



        c->prev_samples[c->prev_samples_pos++] = rlow + rhigh;

        c->prev_samples[c->prev_samples_pos++] = rlow - rhigh;

        ff_g722_apply_qmf(c->prev_samples + c->prev_samples_pos - 24,

                          &xout1, &xout2);

        out_buf[out_len++] = av_clip_int16(xout1 >> 12);

        out_buf[out_len++] = av_clip_int16(xout2 >> 12);

        if (c->prev_samples_pos >= PREV_SAMPLES_BUF_SIZE) {

            memmove(c->prev_samples, c->prev_samples + c->prev_samples_pos - 22,

                    22 * sizeof(c->prev_samples[0]));

            c->prev_samples_pos = 22;

        }

    }

    *data_size = out_len << 1;

    return avpkt->size;

}
