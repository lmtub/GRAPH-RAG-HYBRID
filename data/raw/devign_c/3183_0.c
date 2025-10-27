static int decode_frame(AVCodecContext *avctx, const uint8_t *databuf,

                        float **out_samples)

{

    ATRAC3Context *q = avctx->priv_data;

    int ret, i;

    uint8_t *ptr1;



    if (q->coding_mode == JOINT_STEREO) {

        /* channel coupling mode */

        /* decode Sound Unit 1 */

        init_get_bits(&q->gb, databuf, avctx->block_align * 8);



        ret = decode_channel_sound_unit(q, &q->gb, q->units, out_samples[0], 0,

                                        JOINT_STEREO);

        if (ret != 0)

            return ret;



        /* Framedata of the su2 in the joint-stereo mode is encoded in

         * reverse byte order so we need to swap it first. */

        if (databuf == q->decoded_bytes_buffer) {

            uint8_t *ptr2 = q->decoded_bytes_buffer + avctx->block_align - 1;

            ptr1          = q->decoded_bytes_buffer;

            for (i = 0; i < avctx->block_align / 2; i++, ptr1++, ptr2--)

                FFSWAP(uint8_t, *ptr1, *ptr2);

        } else {

            const uint8_t *ptr2 = databuf + avctx->block_align - 1;

            for (i = 0; i < avctx->block_align; i++)

                q->decoded_bytes_buffer[i] = *ptr2--;

        }



        /* Skip the sync codes (0xF8). */

        ptr1 = q->decoded_bytes_buffer;

        for (i = 4; *ptr1 == 0xF8; i++, ptr1++) {

            if (i >= avctx->block_align)

                return AVERROR_INVALIDDATA;

        }





        /* set the bitstream reader at the start of the second Sound Unit*/

        init_get_bits8(&q->gb, ptr1, q->decoded_bytes_buffer + avctx->block_align - ptr1);



        /* Fill the Weighting coeffs delay buffer */

        memmove(q->weighting_delay, &q->weighting_delay[2],

                4 * sizeof(*q->weighting_delay));

        q->weighting_delay[4] = get_bits1(&q->gb);

        q->weighting_delay[5] = get_bits(&q->gb, 3);



        for (i = 0; i < 4; i++) {

            q->matrix_coeff_index_prev[i] = q->matrix_coeff_index_now[i];

            q->matrix_coeff_index_now[i]  = q->matrix_coeff_index_next[i];

            q->matrix_coeff_index_next[i] = get_bits(&q->gb, 2);

        }



        /* Decode Sound Unit 2. */

        ret = decode_channel_sound_unit(q, &q->gb, &q->units[1],

                                        out_samples[1], 1, JOINT_STEREO);

        if (ret != 0)

            return ret;



        /* Reconstruct the channel coefficients. */

        reverse_matrixing(out_samples[0], out_samples[1],

                          q->matrix_coeff_index_prev,

                          q->matrix_coeff_index_now);



        channel_weighting(out_samples[0], out_samples[1], q->weighting_delay);

    } else {

        /* single channels */

        /* Decode the channel sound units. */

        for (i = 0; i < avctx->channels; i++) {

            /* Set the bitstream reader at the start of a channel sound unit. */

            init_get_bits(&q->gb,

                          databuf + i * avctx->block_align / avctx->channels,

                          avctx->block_align * 8 / avctx->channels);



            ret = decode_channel_sound_unit(q, &q->gb, &q->units[i],

                                            out_samples[i], i, q->coding_mode);

            if (ret != 0)

                return ret;

        }

    }



    /* Apply the iQMF synthesis filter. */

    for (i = 0; i < avctx->channels; i++) {

        float *p1 = out_samples[i];

        float *p2 = p1 + 256;

        float *p3 = p2 + 256;

        float *p4 = p3 + 256;

        ff_atrac_iqmf(p1, p2, 256, p1, q->units[i].delay_buf1, q->temp_buf);

        ff_atrac_iqmf(p4, p3, 256, p3, q->units[i].delay_buf2, q->temp_buf);

        ff_atrac_iqmf(p1, p3, 512, p1, q->units[i].delay_buf3, q->temp_buf);

    }



    return 0;

}
