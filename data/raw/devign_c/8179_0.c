static int decode_channel_sound_unit(ATRAC3Context *q, GetBitContext *gb,

                                     ChannelUnit *snd, float *output,

                                     int channel_num, int coding_mode)

{

    int band, ret, num_subbands, last_tonal, num_bands;

    GainBlock *gain1 = &snd->gain_block[    snd->gc_blk_switch];

    GainBlock *gain2 = &snd->gain_block[1 - snd->gc_blk_switch];



    if (coding_mode == JOINT_STEREO && channel_num == 1) {

        if (get_bits(gb, 2) != 3) {

            av_log(NULL,AV_LOG_ERROR,"JS mono Sound Unit id != 3.\n");

            return AVERROR_INVALIDDATA;

        }

    } else {

        if (get_bits(gb, 6) != 0x28) {

            av_log(NULL,AV_LOG_ERROR,"Sound Unit id != 0x28.\n");

            return AVERROR_INVALIDDATA;

        }

    }



    /* number of coded QMF bands */

    snd->bands_coded = get_bits(gb, 2);



    ret = decode_gain_control(gb, gain2, snd->bands_coded);

    if (ret)

        return ret;



    snd->num_components = decode_tonal_components(gb, snd->components,

                                                  snd->bands_coded);

    if (snd->num_components < 0)

        return snd->num_components;



    num_subbands = decode_spectrum(gb, snd->spectrum);



    /* Merge the decoded spectrum and tonal components. */

    last_tonal = add_tonal_components(snd->spectrum, snd->num_components,

                                      snd->components);





    /* calculate number of used MLT/QMF bands according to the amount of coded

       spectral lines */

    num_bands = (subband_tab[num_subbands] - 1) >> 8;

    if (last_tonal >= 0)

        num_bands = FFMAX((last_tonal + 256) >> 8, num_bands);





    /* Reconstruct time domain samples. */

    for (band = 0; band < 4; band++) {

        /* Perform the IMDCT step without overlapping. */

        if (band <= num_bands)

            imlt(q, &snd->spectrum[band * 256], snd->imdct_buf, band & 1);

        else

            memset(snd->imdct_buf, 0, 512 * sizeof(*snd->imdct_buf));



        /* gain compensation and overlapping */

        ff_atrac_gain_compensation(&q->gainc_ctx, snd->imdct_buf,

                                   &snd->prev_frame[band * 256],

                                   &gain1->g_block[band], &gain2->g_block[band],

                                   256, &output[band * 256]);

    }



    /* Swap the gain control buffers for the next frame. */

    snd->gc_blk_switch ^= 1;



    return 0;

}
