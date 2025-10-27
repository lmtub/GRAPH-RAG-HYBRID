static void apply_channel_coupling(AC3EncodeContext *s)

{

    LOCAL_ALIGNED_16(CoefType, cpl_coords,      [AC3_MAX_BLOCKS], [AC3_MAX_CHANNELS][16]);

#if CONFIG_AC3ENC_FLOAT

    LOCAL_ALIGNED_16(int32_t, fixed_cpl_coords, [AC3_MAX_BLOCKS], [AC3_MAX_CHANNELS][16]);

#else

    int32_t (*fixed_cpl_coords)[AC3_MAX_CHANNELS][16] = cpl_coords;

#endif

    int av_uninit(blk), ch, bnd, i, j;

    CoefSumType energy[AC3_MAX_BLOCKS][AC3_MAX_CHANNELS][16] = {{{0}}};

    int cpl_start, num_cpl_coefs;



    memset(cpl_coords,       0, AC3_MAX_BLOCKS * sizeof(*cpl_coords));

#if CONFIG_AC3ENC_FLOAT

    memset(fixed_cpl_coords, 0, AC3_MAX_BLOCKS * sizeof(*cpl_coords));

#endif



    /* align start to 16-byte boundary. align length to multiple of 32.

        note: coupling start bin % 4 will always be 1 */

    cpl_start     = s->start_freq[CPL_CH] - 1;

    num_cpl_coefs = FFALIGN(s->num_cpl_subbands * 12 + 1, 32);

    cpl_start     = FFMIN(256, cpl_start + num_cpl_coefs) - num_cpl_coefs;



    /* calculate coupling channel from fbw channels */

    for (blk = 0; blk < s->num_blocks; blk++) {

        AC3Block *block = &s->blocks[blk];

        CoefType *cpl_coef = &block->mdct_coef[CPL_CH][cpl_start];

        if (!block->cpl_in_use)

            continue;

        memset(cpl_coef, 0, num_cpl_coefs * sizeof(*cpl_coef));

        for (ch = 1; ch <= s->fbw_channels; ch++) {

            CoefType *ch_coef = &block->mdct_coef[ch][cpl_start];

            if (!block->channel_in_cpl[ch])

                continue;

            for (i = 0; i < num_cpl_coefs; i++)

                cpl_coef[i] += ch_coef[i];

        }



        /* coefficients must be clipped in order to be encoded */

        clip_coefficients(&s->adsp, cpl_coef, num_cpl_coefs);

    }



    /* calculate energy in each band in coupling channel and each fbw channel */

    /* TODO: possibly use SIMD to speed up energy calculation */

    bnd = 0;

    i = s->start_freq[CPL_CH];

    while (i < s->cpl_end_freq) {

        int band_size = s->cpl_band_sizes[bnd];

        for (ch = CPL_CH; ch <= s->fbw_channels; ch++) {

            for (blk = 0; blk < s->num_blocks; blk++) {

                AC3Block *block = &s->blocks[blk];

                if (!block->cpl_in_use || (ch > CPL_CH && !block->channel_in_cpl[ch]))

                    continue;

                for (j = 0; j < band_size; j++) {

                    CoefType v = block->mdct_coef[ch][i+j];

                    MAC_COEF(energy[blk][ch][bnd], v, v);

                }

            }

        }

        i += band_size;

        bnd++;

    }



    /* calculate coupling coordinates for all blocks for all channels */

    for (blk = 0; blk < s->num_blocks; blk++) {

        AC3Block *block  = &s->blocks[blk];

        if (!block->cpl_in_use)

            continue;

        for (ch = 1; ch <= s->fbw_channels; ch++) {

            if (!block->channel_in_cpl[ch])

                continue;

            for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                cpl_coords[blk][ch][bnd] = calc_cpl_coord(energy[blk][ch][bnd],

                                                          energy[blk][CPL_CH][bnd]);

            }

        }

    }



    /* determine which blocks to send new coupling coordinates for */

    for (blk = 0; blk < s->num_blocks; blk++) {

        AC3Block *block  = &s->blocks[blk];

        AC3Block *block0 = blk ? &s->blocks[blk-1] : NULL;



        memset(block->new_cpl_coords, 0, sizeof(block->new_cpl_coords));



        if (block->cpl_in_use) {

            /* send new coordinates if this is the first block, if previous

             * block did not use coupling but this block does, the channels

             * using coupling has changed from the previous block, or the

             * coordinate difference from the last block for any channel is

             * greater than a threshold value. */

            if (blk == 0 || !block0->cpl_in_use) {

                for (ch = 1; ch <= s->fbw_channels; ch++)

                    block->new_cpl_coords[ch] = 1;

            } else {

                for (ch = 1; ch <= s->fbw_channels; ch++) {

                    if (!block->channel_in_cpl[ch])

                        continue;

                    if (!block0->channel_in_cpl[ch]) {

                        block->new_cpl_coords[ch] = 1;

                    } else {

                        CoefSumType coord_diff = 0;

                        for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                            coord_diff += FFABS(cpl_coords[blk-1][ch][bnd] -

                                                cpl_coords[blk  ][ch][bnd]);

                        }

                        coord_diff /= s->num_cpl_bands;

                        if (coord_diff > NEW_CPL_COORD_THRESHOLD)

                            block->new_cpl_coords[ch] = 1;

                    }

                }

            }

        }

    }



    /* calculate final coupling coordinates, taking into account reusing of

       coordinates in successive blocks */

    for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

        blk = 0;

        while (blk < s->num_blocks) {

            int av_uninit(blk1);

            AC3Block *block  = &s->blocks[blk];



            if (!block->cpl_in_use) {

                blk++;

                continue;

            }



            for (ch = 1; ch <= s->fbw_channels; ch++) {

                CoefSumType energy_ch, energy_cpl;

                if (!block->channel_in_cpl[ch])

                    continue;

                energy_cpl = energy[blk][CPL_CH][bnd];

                energy_ch = energy[blk][ch][bnd];

                blk1 = blk+1;

                while (!s->blocks[blk1].new_cpl_coords[ch] && blk1 < s->num_blocks) {

                    if (s->blocks[blk1].cpl_in_use) {

                        energy_cpl += energy[blk1][CPL_CH][bnd];

                        energy_ch += energy[blk1][ch][bnd];

                    }

                    blk1++;

                }

                cpl_coords[blk][ch][bnd] = calc_cpl_coord(energy_ch, energy_cpl);

            }

            blk = blk1;

        }

    }



    /* calculate exponents/mantissas for coupling coordinates */

    for (blk = 0; blk < s->num_blocks; blk++) {

        AC3Block *block = &s->blocks[blk];

        if (!block->cpl_in_use)

            continue;



#if CONFIG_AC3ENC_FLOAT

        s->ac3dsp.float_to_fixed24(fixed_cpl_coords[blk][1],

                                   cpl_coords[blk][1],

                                   s->fbw_channels * 16);

#endif

        s->ac3dsp.extract_exponents(block->cpl_coord_exp[1],

                                    fixed_cpl_coords[blk][1],

                                    s->fbw_channels * 16);



        for (ch = 1; ch <= s->fbw_channels; ch++) {

            int bnd, min_exp, max_exp, master_exp;



            if (!block->new_cpl_coords[ch])

                continue;



            /* determine master exponent */

            min_exp = max_exp = block->cpl_coord_exp[ch][0];

            for (bnd = 1; bnd < s->num_cpl_bands; bnd++) {

                int exp = block->cpl_coord_exp[ch][bnd];

                min_exp = FFMIN(exp, min_exp);

                max_exp = FFMAX(exp, max_exp);

            }

            master_exp = ((max_exp - 15) + 2) / 3;

            master_exp = FFMAX(master_exp, 0);

            while (min_exp < master_exp * 3)

                master_exp--;

            for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                block->cpl_coord_exp[ch][bnd] = av_clip(block->cpl_coord_exp[ch][bnd] -

                                                        master_exp * 3, 0, 15);

            }

            block->cpl_master_exp[ch] = master_exp;



            /* quantize mantissas */

            for (bnd = 0; bnd < s->num_cpl_bands; bnd++) {

                int cpl_exp  = block->cpl_coord_exp[ch][bnd];

                int cpl_mant = (fixed_cpl_coords[blk][ch][bnd] << (5 + cpl_exp + master_exp * 3)) >> 24;

                if (cpl_exp == 15)

                    cpl_mant >>= 1;

                else

                    cpl_mant -= 16;



                block->cpl_coord_mant[ch][bnd] = cpl_mant;

            }

        }

    }



    if (CONFIG_EAC3_ENCODER && s->eac3)

        ff_eac3_set_cpl_states(s);

}
