static int qcelp_decode_frame(AVCodecContext *avctx, void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    QCELPContext *q = avctx->priv_data;

    float *outbuffer = data;

    int   i;

    float quantized_lspf[10], lpc[10];

    float gain[16];

    float *formant_mem;



    if((q->bitrate = determine_bitrate(avctx, buf_size, &buf)) == I_F_Q)

    {

        warn_insufficient_frame_quality(avctx, "bitrate cannot be determined.");

        goto erasure;

    }



    if(q->bitrate == RATE_OCTAVE &&

       (q->first16bits = AV_RB16(buf)) == 0xFFFF)

    {

        warn_insufficient_frame_quality(avctx, "Bitrate is 1/8 and first 16 bits are on.");

        goto erasure;

    }



    if(q->bitrate > SILENCE)

    {

        const QCELPBitmap *bitmaps     = qcelp_unpacking_bitmaps_per_rate[q->bitrate];

        const QCELPBitmap *bitmaps_end = qcelp_unpacking_bitmaps_per_rate[q->bitrate]

                                       + qcelp_unpacking_bitmaps_lengths[q->bitrate];

        uint8_t           *unpacked_data = (uint8_t *)&q->frame;



        init_get_bits(&q->gb, buf, 8*buf_size);



        memset(&q->frame, 0, sizeof(QCELPFrame));



        for(; bitmaps < bitmaps_end; bitmaps++)

            unpacked_data[bitmaps->index] |= get_bits(&q->gb, bitmaps->bitlen) << bitmaps->bitpos;



        // Check for erasures/blanks on rates 1, 1/4 and 1/8.

        if(q->frame.reserved)

        {

            warn_insufficient_frame_quality(avctx, "Wrong data in reserved frame area.");

            goto erasure;

        }

        if(q->bitrate == RATE_QUARTER &&

           codebook_sanity_check_for_rate_quarter(q->frame.cbgain))

        {

            warn_insufficient_frame_quality(avctx, "Codebook gain sanity check failed.");

            goto erasure;

        }



        if(q->bitrate >= RATE_HALF)

        {

            for(i=0; i<4; i++)

            {

                if(q->frame.pfrac[i] && q->frame.plag[i] >= 124)

                {

                    warn_insufficient_frame_quality(avctx, "Cannot initialize pitch filter.");

                    goto erasure;

                }

            }

        }

    }



    decode_gain_and_index(q, gain);

    compute_svector(q, gain, outbuffer);



    if(decode_lspf(q, quantized_lspf) < 0)

    {

        warn_insufficient_frame_quality(avctx, "Badly received packets in frame.");

        goto erasure;

    }





    apply_pitch_filters(q, outbuffer);



    if(q->bitrate == I_F_Q)

    {

erasure:

        q->bitrate = I_F_Q;

        q->erasure_count++;

        decode_gain_and_index(q, gain);

        compute_svector(q, gain, outbuffer);

        decode_lspf(q, quantized_lspf);

        apply_pitch_filters(q, outbuffer);

    }else

        q->erasure_count = 0;



    formant_mem = q->formant_mem + 10;

    for(i=0; i<4; i++)

    {

        interpolate_lpc(q, quantized_lspf, lpc, i);

        ff_celp_lp_synthesis_filterf(formant_mem, lpc, outbuffer + i * 40, 40,

                                     10);

        formant_mem += 40;

    }



    // postfilter, as per TIA/EIA/IS-733 2.4.8.6

    postfilter(q, outbuffer, lpc);



    memcpy(q->formant_mem, q->formant_mem + 160, 10 * sizeof(float));



    memcpy(q->prev_lspf, quantized_lspf, sizeof(q->prev_lspf));

    q->prev_bitrate = q->bitrate;



    *data_size = 160 * sizeof(*outbuffer);



    return buf_size;

}
