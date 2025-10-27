int ff_dca_xll_decode_audio(DCAContext *s, AVFrame *frame)

{

    /* FIXME: Decodes only the first frequency band. */

    int seg, chset_i;



    /* Coding parameters for each channel set. */

    struct coding_params {

        int seg_type;

        int rice_code_flag[16];

        int pancAuxABIT[16];

        int pancABIT0[16];  /* Not sure what this is */

        int pancABIT[16];   /* Not sure what this is */

        int nSamplPart0[16];

    } param_state[16];



    GetBitContext *gb = &s->xll_navi.gb;

    int *history;



    /* Layout: First the sample buffer for one segment per channel,

     * followed by history buffers of DCA_XLL_AORDER_MAX samples for

     * each channel. */

    av_fast_malloc(&s->xll_sample_buf, &s->xll_sample_buf_size,

                   (s->xll_smpl_in_seg + DCA_XLL_AORDER_MAX) *

                   s->xll_channels * sizeof(*s->xll_sample_buf));

    if (!s->xll_sample_buf)

        return AVERROR(ENOMEM);



    history = s->xll_sample_buf + s->xll_smpl_in_seg * s->xll_channels;



    for (seg = 0; seg < s->xll_segments; seg++) {

        unsigned in_channel;



        for (chset_i = in_channel = 0; chset_i < s->xll_nch_sets; chset_i++) {

            /* The spec isn't very explicit, but I think the NAVI sizes are in bytes. */

            int end_pos = get_bits_count(gb) +

                          8 * s->xll_navi.chset_size[0][seg][chset_i];

            int i, j;

            struct coding_params *params = &param_state[chset_i];

            /* I think this flag means that we should keep seg_type and

             * other parameters from the previous segment. */

            int use_seg_state_code_param;

            XllChSetSubHeader *chset = &s->xll_chsets[chset_i];

            if (in_channel >= s->avctx->channels)

                /* FIXME: Could go directly to next segment */

                goto next_chset;



            if (s->avctx->sample_rate != chset->sampling_frequency) {

                av_log(s->avctx, AV_LOG_WARNING,

                       "XLL: unexpected chset sample rate %d, expected %d\n",

                       chset->sampling_frequency, s->avctx->sample_rate);

                goto next_chset;

            }

            if (seg != 0)

                use_seg_state_code_param = get_bits(gb, 1);

            else

                use_seg_state_code_param = 0;



            if (!use_seg_state_code_param) {

                int num_param_sets, i;

                unsigned bits4ABIT;



                params->seg_type = get_bits(gb, 1);

                num_param_sets   = params->seg_type ? 1 : chset->channels;



                if (chset->bit_width > 16) {

                    bits4ABIT = 5;

                } else {

                    if (chset->bit_width > 8)

                        bits4ABIT = 4;

                    else

                        bits4ABIT = 3;

                    if (s->xll_nch_sets > 1)

                        bits4ABIT++;

                }



                for (i = 0; i < num_param_sets; i++) {

                    params->rice_code_flag[i] = get_bits(gb, 1);

                    if (!params->seg_type && params->rice_code_flag[i] && get_bits(gb, 1))

                        params->pancAuxABIT[i] = get_bits(gb, bits4ABIT) + 1;

                    else

                        params->pancAuxABIT[i] = 0;

                }



                for (i = 0; i < num_param_sets; i++) {

                    if (!seg) {

                        /* Parameters for part 1 */

                        params->pancABIT0[i] = get_bits(gb, bits4ABIT);

                        if (params->rice_code_flag[i] == 0 && params->pancABIT0[i] > 0)

                            /* For linear code */

                            params->pancABIT0[i]++;



                        /* NOTE: In the spec, not indexed by band??? */

                        if (params->seg_type == 0)

                            params->nSamplPart0[i] = chset->adapt_order[0][i];

                        else

                            params->nSamplPart0[i] = chset->adapt_order_max[0];

                    } else

                        params->nSamplPart0[i] = 0;



                    /* Parameters for part 2 */

                    params->pancABIT[i] = get_bits(gb, bits4ABIT);

                    if (params->rice_code_flag[i] == 0 && params->pancABIT[i] > 0)

                        /* For linear code */

                        params->pancABIT[i]++;

                }

            }

            for (i = 0; i < chset->channels; i++) {

                int param_index = params->seg_type ? 0 : i;

                int bits        = params->pancABIT0[param_index];

                int part0       = params->nSamplPart0[param_index];

                int *sample_buf = s->xll_sample_buf +

                                  (in_channel + i) * s->xll_smpl_in_seg;



                if (!params->rice_code_flag[param_index]) {

                    /* Linear code */

                    if (bits)

                        for (j = 0; j < part0; j++)

                            sample_buf[j] = get_bits_sm(gb, bits);

                    else

                        memset(sample_buf, 0, part0 * sizeof(sample_buf[0]));



                    /* Second part */

                    bits = params->pancABIT[param_index];

                    if (bits)

                        for (j = part0; j < s->xll_smpl_in_seg; j++)

                            sample_buf[j] = get_bits_sm(gb, bits);

                    else

                        memset(sample_buf + part0, 0,

                               (s->xll_smpl_in_seg - part0) * sizeof(sample_buf[0]));

                } else {

                    int aux_bits = params->pancAuxABIT[param_index];



                    for (j = 0; j < part0; j++) {

                        /* FIXME: Is this identical to Golomb code? */

                        int t = get_unary(gb, 1, 33) << bits;

                        /* FIXME: Could move this test outside of the loop, for efficiency. */

                        if (bits)

                            t |= get_bits(gb, bits);

                        sample_buf[j] = (t & 1) ? -(t >> 1) - 1 : (t >> 1);

                    }



                    /* Second part */

                    bits = params->pancABIT[param_index];



                    /* Follow the spec's suggestion of using the

                     * buffer also to store the hybrid-rice flags. */

                    memset(sample_buf + part0, 0,

                           (s->xll_smpl_in_seg - part0) * sizeof(sample_buf[0]));



                    if (aux_bits > 0) {

                        /* For hybrid rice encoding, some samples are linearly

                         * coded. According to the spec, "nBits4SamplLoci" bits

                         * are used for each index, but this value is not

                         * defined. I guess we should use log2(xll_smpl_in_seg)

                         * bits. */

                        int count = get_bits(gb, s->xll_log_smpl_in_seg);

                        av_log(s->avctx, AV_LOG_DEBUG, "aux count %d (bits %d)\n",

                               count, s->xll_log_smpl_in_seg);



                        for (j = 0; j < count; j++)

                            sample_buf[get_bits(gb, s->xll_log_smpl_in_seg)] = 1;

                    }

                    for (j = part0; j < s->xll_smpl_in_seg; j++) {

                        if (!sample_buf[j]) {

                            int t = get_unary(gb, 1, 33);

                            if (bits)

                                t = (t << bits) | get_bits(gb, bits);

                            sample_buf[j] = (t & 1) ? -(t >> 1) - 1 : (t >> 1);

                        } else

                            sample_buf[j] = get_bits_sm(gb, aux_bits);

                    }

                }

            }



            for (i = 0; i < chset->channels; i++) {

                unsigned adapt_order = chset->adapt_order[0][i];

                int *sample_buf = s->xll_sample_buf +

                                  (in_channel + i) * s->xll_smpl_in_seg;

                int *prev = history + (in_channel + i) * DCA_XLL_AORDER_MAX;



                if (!adapt_order) {

                    unsigned order;

                    for (order = chset->fixed_order[0][i]; order > 0; order--) {

                        unsigned j;

                        for (j = 1; j < s->xll_smpl_in_seg; j++)

                            sample_buf[j] += sample_buf[j - 1];

                    }

                } else

                    /* Inverse adaptive prediction, in place. */

                    dca_xll_inv_adapt_pred(sample_buf, s->xll_smpl_in_seg,

                                           adapt_order, seg ? prev : NULL,

                                           chset->lpc_refl_coeffs_q_ind[0][i]);

                memcpy(prev, sample_buf + s->xll_smpl_in_seg - DCA_XLL_AORDER_MAX,

                       DCA_XLL_AORDER_MAX * sizeof(*prev));

            }

            for (i = 1; i < chset->channels; i += 2) {

                int coeff = chset->pw_ch_pairs_coeffs[0][i / 2];

                if (coeff != 0) {

                    int *sample_buf = s->xll_sample_buf +

                                      (in_channel + i) * s->xll_smpl_in_seg;

                    int *prev = sample_buf - s->xll_smpl_in_seg;

                    unsigned j;

                    for (j = 0; j < s->xll_smpl_in_seg; j++)

                        /* Shift is unspecified, but should apparently be 3. */

                        sample_buf[j] += ((int64_t) coeff * prev[j] + 4) >> 3;

                }

            }



            if (s->xll_scalable_lsb) {

                int lsb_start = end_pos - 8 * chset->lsb_fsize[0] -

                                8 * (s->xll_banddata_crc & 2);

                int done;

                i = get_bits_count(gb);

                if (i > lsb_start) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "chset data lsb exceeds NAVI size, end_pos %d, lsb_start %d, pos %d\n",

                           end_pos, lsb_start, i);

                    return AVERROR_INVALIDDATA;

                }

                if (i < lsb_start)

                    skip_bits_long(gb, lsb_start - i);



                for (i = done = 0; i < chset->channels; i++) {

                    int bits = chset->scalable_lsbs[0][i];

                    if (bits > 0) {

                        /* The channel reordering is conceptually done

                         * before adding the lsb:s, so we need to do

                         * the inverse permutation here. */

                        unsigned pi = chset->orig_chan_order_inv[0][i];

                        int *sample_buf = s->xll_sample_buf +

                                          (in_channel + pi) * s->xll_smpl_in_seg;

                        int adj = chset->bit_width_adj_per_ch[0][i];

                        int msb_shift = bits;

                        unsigned j;



                        if (adj > 0)

                            msb_shift += adj - 1;



                        for (j = 0; j < s->xll_smpl_in_seg; j++)

                            sample_buf[j] = (sample_buf[j] << msb_shift) +

                                            (get_bits(gb, bits) << adj);



                        done += bits * s->xll_smpl_in_seg;

                    }

                }

                if (done > 8 * chset->lsb_fsize[0]) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "chset lsb exceeds lsb_size\n");

                    return AVERROR_INVALIDDATA;

                }

            }



            /* Store output. */

            for (i = 0; i < chset->channels; i++) {

                int *sample_buf = s->xll_sample_buf +

                                  (in_channel + i) * s->xll_smpl_in_seg;

                int shift = 1 - chset->bit_resolution;

                int out_channel = chset->orig_chan_order[0][i];

                float *out;



                /* XLL uses the channel order C, L, R, and we want L,

                 * R, C. FIXME: Generalize. */

                if (chset->ch_mask_enabled &&

                    (chset->ch_mask & 7) == 7 && out_channel < 3)

                    out_channel = out_channel ? out_channel - 1 : 2;



                out_channel += in_channel;

                if (out_channel >= s->avctx->channels)

                    continue;



                out  = (float *) frame->extended_data[out_channel];

                out += seg * s->xll_smpl_in_seg;



                /* NOTE: A one bit means residual encoding is *not* used. */

                if ((chset->residual_encode >> i) & 1) {

                    /* Replace channel samples.

                     * FIXME: Most likely not the right thing to do. */

                    for (j = 0; j < s->xll_smpl_in_seg; j++)

                        out[j] = ldexpf(sample_buf[j], shift);

                } else {

                    /* Add residual signal to core channel */

                    for (j = 0; j < s->xll_smpl_in_seg; j++)

                        out[j] += ldexpf(sample_buf[j], shift);

                }

            }



            if (chset->downmix_coeff_code_embedded &&

                !chset->primary_ch_set && chset->hier_chset) {

                /* Undo hierarchical downmix of earlier channels. */

                unsigned mix_channel;

                for (mix_channel = 0; mix_channel < in_channel; mix_channel++) {

                    float *mix_buf;

                    const int *col;

                    float coeff;

                    unsigned row;

                    /* Similar channel reorder C, L, R vs L, R, C reorder. */

                    if (chset->ch_mask_enabled &&

                        (chset->ch_mask & 7) == 7 && mix_channel < 3)

                        mix_buf = (float *) frame->extended_data[mix_channel ? mix_channel - 1 : 2];

                    else

                        mix_buf = (float *) frame->extended_data[mix_channel];



                    mix_buf += seg * s->xll_smpl_in_seg;

                    col = &chset->downmix_coeffs[mix_channel * (chset->channels + 1)];



                    /* Scale */

                    coeff = ldexpf(col[0], -16);

                    for (j = 0; j < s->xll_smpl_in_seg; j++)

                        mix_buf[j] *= coeff;



                    for (row = 0;

                         row < chset->channels && in_channel + row < s->avctx->channels;

                         row++)

                        if (col[row + 1]) {

                            const float *new_channel =

                                (const float *) frame->extended_data[in_channel + row];

                            new_channel += seg * s->xll_smpl_in_seg;

                            coeff        = ldexpf(col[row + 1], -15);

                            for (j = 0; j < s->xll_smpl_in_seg; j++)

                                mix_buf[j] -= coeff * new_channel[j];

                        }

                }

            }



next_chset:

            in_channel += chset->channels;

            /* Skip to next channel set using the NAVI info. */

            i = get_bits_count(gb);

            if (i > end_pos) {

                av_log(s->avctx, AV_LOG_ERROR,

                       "chset data exceeds NAVI size\n");

                return AVERROR_INVALIDDATA;

            }

            if (i < end_pos)

                skip_bits_long(gb, end_pos - i);

        }

    }

    return 0;

}
