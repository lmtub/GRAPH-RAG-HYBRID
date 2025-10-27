static int synth_frame(AVCodecContext *ctx, GetBitContext *gb, int frame_idx,

                       float *samples,

                       const double *lsps, const double *prev_lsps,

                       float *excitation, float *synth)

{

    WMAVoiceContext *s = ctx->priv_data;

    int n, n_blocks_x2, log_n_blocks_x2, cur_pitch_val;

    int pitch[MAX_BLOCKS], last_block_pitch;



    /* Parse frame type ("frame header"), see frame_descs */

    int bd_idx = s->vbm_tree[get_vlc2(gb, frame_type_vlc.table, 6, 3)], block_nsamples;



    if (bd_idx < 0) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid frame type VLC code, skipping\n");

        return -1;

    }



    block_nsamples = MAX_FRAMESIZE / frame_descs[bd_idx].n_blocks;



    /* Pitch calculation for ACB_TYPE_ASYMMETRIC ("pitch-per-frame") */

    if (frame_descs[bd_idx].acb_type == ACB_TYPE_ASYMMETRIC) {

        /* Pitch is provided per frame, which is interpreted as the pitch of

         * the last sample of the last block of this frame. We can interpolate

         * the pitch of other blocks (and even pitch-per-sample) by gradually

         * incrementing/decrementing prev_frame_pitch to cur_pitch_val. */

        n_blocks_x2      = frame_descs[bd_idx].n_blocks << 1;

        log_n_blocks_x2  = frame_descs[bd_idx].log_n_blocks + 1;

        cur_pitch_val    = s->min_pitch_val + get_bits(gb, s->pitch_nbits);

        cur_pitch_val    = FFMIN(cur_pitch_val, s->max_pitch_val - 1);

        if (s->last_acb_type == ACB_TYPE_NONE ||

            20 * abs(cur_pitch_val - s->last_pitch_val) >

                (cur_pitch_val + s->last_pitch_val))

            s->last_pitch_val = cur_pitch_val;



        /* pitch per block */

        for (n = 0; n < frame_descs[bd_idx].n_blocks; n++) {

            int fac = n * 2 + 1;



            pitch[n] = (MUL16(fac,                 cur_pitch_val) +

                        MUL16((n_blocks_x2 - fac), s->last_pitch_val) +

                        frame_descs[bd_idx].n_blocks) >> log_n_blocks_x2;

        }



        /* "pitch-diff-per-sample" for calculation of pitch per sample */

        s->pitch_diff_sh16 =

            ((cur_pitch_val - s->last_pitch_val) << 16) / MAX_FRAMESIZE;

    }



    /* Global gain (if silence) and pitch-adaptive window coordinates */

    switch (frame_descs[bd_idx].fcb_type) {

    case FCB_TYPE_SILENCE:

        s->silence_gain = wmavoice_gain_silence[get_bits(gb, 8)];

        break;

    case FCB_TYPE_AW_PULSES:

        aw_parse_coords(s, gb, pitch);

        break;

    }



    for (n = 0; n < frame_descs[bd_idx].n_blocks; n++) {

        int bl_pitch_sh2;



        /* Pitch calculation for ACB_TYPE_HAMMING ("pitch-per-block") */

        switch (frame_descs[bd_idx].acb_type) {

        case ACB_TYPE_HAMMING: {

            /* Pitch is given per block. Per-block pitches are encoded as an

             * absolute value for the first block, and then delta values

             * relative to this value) for all subsequent blocks. The scale of

             * this pitch value is semi-logaritmic compared to its use in the

             * decoder, so we convert it to normal scale also. */

            int block_pitch,

                t1 = (s->block_conv_table[1] - s->block_conv_table[0]) << 2,

                t2 = (s->block_conv_table[2] - s->block_conv_table[1]) << 1,

                t3 =  s->block_conv_table[3] - s->block_conv_table[2] + 1;



            if (n == 0) {

                block_pitch = get_bits(gb, s->block_pitch_nbits);

            } else

                block_pitch = last_block_pitch - s->block_delta_pitch_hrange +

                                 get_bits(gb, s->block_delta_pitch_nbits);

            /* Convert last_ so that any next delta is within _range */

            last_block_pitch = av_clip(block_pitch,

                                       s->block_delta_pitch_hrange,

                                       s->block_pitch_range -

                                           s->block_delta_pitch_hrange);



            /* Convert semi-log-style scale back to normal scale */

            if (block_pitch < t1) {

                bl_pitch_sh2 = (s->block_conv_table[0] << 2) + block_pitch;

            } else {

                block_pitch -= t1;

                if (block_pitch < t2) {

                    bl_pitch_sh2 =

                        (s->block_conv_table[1] << 2) + (block_pitch << 1);

                } else {

                    block_pitch -= t2;

                    if (block_pitch < t3) {

                        bl_pitch_sh2 =

                            (s->block_conv_table[2] + block_pitch) << 2;

                    } else

                        bl_pitch_sh2 = s->block_conv_table[3] << 2;

                }

            }

            pitch[n] = bl_pitch_sh2 >> 2;

            break;

        }



        case ACB_TYPE_ASYMMETRIC: {

            bl_pitch_sh2 = pitch[n] << 2;

            break;

        }



        default: // ACB_TYPE_NONE has no pitch

            bl_pitch_sh2 = 0;

            break;

        }



        synth_block(s, gb, n, block_nsamples, bl_pitch_sh2,

                    lsps, prev_lsps, &frame_descs[bd_idx],

                    &excitation[n * block_nsamples],

                    &synth[n * block_nsamples]);

    }



    /* Averaging projection filter, if applicable. Else, just copy samples

     * from synthesis buffer */

    if (s->do_apf) {

        double i_lsps[MAX_LSPS];

        float lpcs[MAX_LSPS];



        for (n = 0; n < s->lsps; n++) // LSF -> LSP

            i_lsps[n] = cos(0.5 * (prev_lsps[n] + lsps[n]));

        ff_acelp_lspd2lpc(i_lsps, lpcs, s->lsps >> 1);

        postfilter(s, synth, samples, 80, lpcs,

                   &s->zero_exc_pf[s->history_nsamples + MAX_FRAMESIZE * frame_idx],

                   frame_descs[bd_idx].fcb_type, pitch[0]);



        for (n = 0; n < s->lsps; n++) // LSF -> LSP

            i_lsps[n] = cos(lsps[n]);

        ff_acelp_lspd2lpc(i_lsps, lpcs, s->lsps >> 1);

        postfilter(s, &synth[80], &samples[80], 80, lpcs,

                   &s->zero_exc_pf[s->history_nsamples + MAX_FRAMESIZE * frame_idx + 80],

                   frame_descs[bd_idx].fcb_type, pitch[0]);

    } else

        memcpy(samples, synth, 160 * sizeof(synth[0]));



    /* Cache values for next frame */

    s->frame_cntr++;

    if (s->frame_cntr >= 0xFFFF) s->frame_cntr -= 0xFFFF; // i.e. modulo (%)

    s->last_acb_type = frame_descs[bd_idx].acb_type;

    switch (frame_descs[bd_idx].acb_type) {

    case ACB_TYPE_NONE:

        s->last_pitch_val = 0;

        break;

    case ACB_TYPE_ASYMMETRIC:

        s->last_pitch_val = cur_pitch_val;

        break;

    case ACB_TYPE_HAMMING:

        s->last_pitch_val = pitch[frame_descs[bd_idx].n_blocks - 1];

        break;

    }



    return 0;

}
