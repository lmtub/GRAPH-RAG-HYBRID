int ff_eac3_parse_header(AC3DecodeContext *s)

{

    int i, blk, ch;

    int ac3_exponent_strategy, parse_aht_info, parse_spx_atten_data;

    int parse_transient_proc_info;

    int num_cpl_blocks;

    GetBitContext *gbc = &s->gbc;



    /* An E-AC-3 stream can have multiple independent streams which the

       application can select from. each independent stream can also contain

       dependent streams which are used to add or replace channels. */

    if (s->frame_type == EAC3_FRAME_TYPE_DEPENDENT) {

        avpriv_request_sample(s->avctx, "Dependent substream decoding");

        return AAC_AC3_PARSE_ERROR_FRAME_TYPE;

    } else if (s->frame_type == EAC3_FRAME_TYPE_RESERVED) {

        av_log(s->avctx, AV_LOG_ERROR, "Reserved frame type\n");

        return AAC_AC3_PARSE_ERROR_FRAME_TYPE;

    }



    /* The substream id indicates which substream this frame belongs to. each

       independent stream has its own substream id, and the dependent streams

       associated to an independent stream have matching substream id's. */

    if (s->substreamid) {

        /* only decode substream with id=0. skip any additional substreams. */

        avpriv_request_sample(s->avctx, "Additional substreams");

        return AAC_AC3_PARSE_ERROR_FRAME_TYPE;

    }



    if (s->bit_alloc_params.sr_code == EAC3_SR_CODE_REDUCED) {

        /* The E-AC-3 specification does not tell how to handle reduced sample

           rates in bit allocation.  The best assumption would be that it is

           handled like AC-3 DolbyNet, but we cannot be sure until we have a

           sample which utilizes this feature. */

        avpriv_request_sample(s->avctx, "Reduced sampling rate");

        return AVERROR_PATCHWELCOME;

    }

    skip_bits(gbc, 5); // skip bitstream id



    /* volume control params */

    for (i = 0; i < (s->channel_mode ? 1 : 2); i++) {

        skip_bits(gbc, 5); // skip dialog normalization

        if (get_bits1(gbc)) {

            skip_bits(gbc, 8); // skip compression gain word

        }

    }



    /* dependent stream channel map */

    if (s->frame_type == EAC3_FRAME_TYPE_DEPENDENT) {

        if (get_bits1(gbc)) {

            skip_bits(gbc, 16); // skip custom channel map

        }

    }



    /* mixing metadata */

    if (get_bits1(gbc)) {

        /* center and surround mix levels */

        if (s->channel_mode > AC3_CHMODE_STEREO) {

            s->preferred_downmix = get_bits(gbc, 2);

            if (s->channel_mode & 1) {

                /* if three front channels exist */

                s->center_mix_level_ltrt = get_bits(gbc, 3);

                s->center_mix_level      = get_bits(gbc, 3);

            }

            if (s->channel_mode & 4) {

                /* if a surround channel exists */

                s->surround_mix_level_ltrt = av_clip(get_bits(gbc, 3), 3, 7);

                s->surround_mix_level      = av_clip(get_bits(gbc, 3), 3, 7);

            }

        }



        /* lfe mix level */

        if (s->lfe_on && (s->lfe_mix_level_exists = get_bits1(gbc))) {

            s->lfe_mix_level = get_bits(gbc, 5);

        }



        /* info for mixing with other streams and substreams */

        if (s->frame_type == EAC3_FRAME_TYPE_INDEPENDENT) {

            for (i = 0; i < (s->channel_mode ? 1 : 2); i++) {

                // TODO: apply program scale factor

                if (get_bits1(gbc)) {

                    skip_bits(gbc, 6);  // skip program scale factor

                }

            }

            if (get_bits1(gbc)) {

                skip_bits(gbc, 6);  // skip external program scale factor

            }

            /* skip mixing parameter data */

            switch(get_bits(gbc, 2)) {

                case 1: skip_bits(gbc, 5);  break;

                case 2: skip_bits(gbc, 12); break;

                case 3: {

                    int mix_data_size = (get_bits(gbc, 5) + 2) << 3;

                    skip_bits_long(gbc, mix_data_size);

                    break;

                }

            }

            /* skip pan information for mono or dual mono source */

            if (s->channel_mode < AC3_CHMODE_STEREO) {

                for (i = 0; i < (s->channel_mode ? 1 : 2); i++) {

                    if (get_bits1(gbc)) {

                        /* note: this is not in the ATSC A/52B specification

                           reference: ETSI TS 102 366 V1.1.1

                                      section: E.1.3.1.25 */

                        skip_bits(gbc, 8);  // skip pan mean direction index

                        skip_bits(gbc, 6);  // skip reserved paninfo bits

                    }

                }

            }

            /* skip mixing configuration information */

            if (get_bits1(gbc)) {

                for (blk = 0; blk < s->num_blocks; blk++) {

                    if (s->num_blocks == 1 || get_bits1(gbc)) {

                        skip_bits(gbc, 5);

                    }

                }

            }

        }

    }



    /* informational metadata */

    if (get_bits1(gbc)) {

        s->bitstream_mode = get_bits(gbc, 3);

        skip_bits(gbc, 2); // skip copyright bit and original bitstream bit

        if (s->channel_mode == AC3_CHMODE_STEREO) {

            s->dolby_surround_mode  = get_bits(gbc, 2);

            s->dolby_headphone_mode = get_bits(gbc, 2);

        }

        if (s->channel_mode >= AC3_CHMODE_2F2R) {

            s->dolby_surround_ex_mode = get_bits(gbc, 2);

        }

        for (i = 0; i < (s->channel_mode ? 1 : 2); i++) {

            if (get_bits1(gbc)) {

                skip_bits(gbc, 8); // skip mix level, room type, and A/D converter type

            }

        }

        if (s->bit_alloc_params.sr_code != EAC3_SR_CODE_REDUCED) {

            skip_bits1(gbc); // skip source sample rate code

        }

    }



    /* converter synchronization flag

       If frames are less than six blocks, this bit should be turned on

       once every 6 blocks to indicate the start of a frame set.

       reference: RFC 4598, Section 2.1.3  Frame Sets */

    if (s->frame_type == EAC3_FRAME_TYPE_INDEPENDENT && s->num_blocks != 6) {

        skip_bits1(gbc); // skip converter synchronization flag

    }



    /* original frame size code if this stream was converted from AC-3 */

    if (s->frame_type == EAC3_FRAME_TYPE_AC3_CONVERT &&

            (s->num_blocks == 6 || get_bits1(gbc))) {

        skip_bits(gbc, 6); // skip frame size code

    }



    /* additional bitstream info */

    if (get_bits1(gbc)) {

        int addbsil = get_bits(gbc, 6);

        for (i = 0; i < addbsil + 1; i++) {

            skip_bits(gbc, 8); // skip additional bit stream info

        }

    }



    /* audio frame syntax flags, strategy data, and per-frame data */



    if (s->num_blocks == 6) {

        ac3_exponent_strategy = get_bits1(gbc);

        parse_aht_info        = get_bits1(gbc);

    } else {

        /* less than 6 blocks, so use AC-3-style exponent strategy syntax, and

           do not use AHT */

        ac3_exponent_strategy = 1;

        parse_aht_info = 0;

    }



    s->snr_offset_strategy    = get_bits(gbc, 2);

    parse_transient_proc_info = get_bits1(gbc);



    s->block_switch_syntax = get_bits1(gbc);

    if (!s->block_switch_syntax)

        memset(s->block_switch, 0, sizeof(s->block_switch));



    s->dither_flag_syntax = get_bits1(gbc);

    if (!s->dither_flag_syntax) {

        for (ch = 1; ch <= s->fbw_channels; ch++)

            s->dither_flag[ch] = 1;

    }

    s->dither_flag[CPL_CH] = s->dither_flag[s->lfe_ch] = 0;



    s->bit_allocation_syntax = get_bits1(gbc);

    if (!s->bit_allocation_syntax) {

        /* set default bit allocation parameters */

        s->bit_alloc_params.slow_decay = ff_ac3_slow_decay_tab[2];

        s->bit_alloc_params.fast_decay = ff_ac3_fast_decay_tab[1];

        s->bit_alloc_params.slow_gain  = ff_ac3_slow_gain_tab [1];

        s->bit_alloc_params.db_per_bit = ff_ac3_db_per_bit_tab[2];

        s->bit_alloc_params.floor      = ff_ac3_floor_tab     [7];

    }



    s->fast_gain_syntax  = get_bits1(gbc);

    s->dba_syntax        = get_bits1(gbc);

    s->skip_syntax       = get_bits1(gbc);

    parse_spx_atten_data = get_bits1(gbc);



    /* coupling strategy occurrence and coupling use per block */

    num_cpl_blocks = 0;

    if (s->channel_mode > 1) {

        for (blk = 0; blk < s->num_blocks; blk++) {

            s->cpl_strategy_exists[blk] = (!blk || get_bits1(gbc));

            if (s->cpl_strategy_exists[blk]) {

                s->cpl_in_use[blk] = get_bits1(gbc);

            } else {

                s->cpl_in_use[blk] = s->cpl_in_use[blk-1];

            }

            num_cpl_blocks += s->cpl_in_use[blk];

        }

    } else {

        memset(s->cpl_in_use, 0, sizeof(s->cpl_in_use));

    }



    /* exponent strategy data */

    if (ac3_exponent_strategy) {

        /* AC-3-style exponent strategy syntax */

        for (blk = 0; blk < s->num_blocks; blk++) {

            for (ch = !s->cpl_in_use[blk]; ch <= s->fbw_channels; ch++) {

                s->exp_strategy[blk][ch] = get_bits(gbc, 2);

            }

        }

    } else {

        /* LUT-based exponent strategy syntax */

        for (ch = !((s->channel_mode > 1) && num_cpl_blocks); ch <= s->fbw_channels; ch++) {

            int frmchexpstr = get_bits(gbc, 5);

            for (blk = 0; blk < 6; blk++) {

                s->exp_strategy[blk][ch] = ff_eac3_frm_expstr[frmchexpstr][blk];

            }

        }

    }

    /* LFE exponent strategy */

    if (s->lfe_on) {

        for (blk = 0; blk < s->num_blocks; blk++) {

            s->exp_strategy[blk][s->lfe_ch] = get_bits1(gbc);

        }

    }

    /* original exponent strategies if this stream was converted from AC-3 */

    if (s->frame_type == EAC3_FRAME_TYPE_INDEPENDENT &&

            (s->num_blocks == 6 || get_bits1(gbc))) {

        skip_bits(gbc, 5 * s->fbw_channels); // skip converter channel exponent strategy

    }



    /* determine which channels use AHT */

    if (parse_aht_info) {

        /* For AHT to be used, all non-zero blocks must reuse exponents from

           the first block.  Furthermore, for AHT to be used in the coupling

           channel, all blocks must use coupling and use the same coupling

           strategy. */

        s->channel_uses_aht[CPL_CH]=0;

        for (ch = (num_cpl_blocks != 6); ch <= s->channels; ch++) {

            int use_aht = 1;

            for (blk = 1; blk < 6; blk++) {

                if ((s->exp_strategy[blk][ch] != EXP_REUSE) ||

                        (!ch && s->cpl_strategy_exists[blk])) {

                    use_aht = 0;

                    break;

                }

            }

            s->channel_uses_aht[ch] = use_aht && get_bits1(gbc);

        }

    } else {

        memset(s->channel_uses_aht, 0, sizeof(s->channel_uses_aht));

    }



    /* per-frame SNR offset */

    if (!s->snr_offset_strategy) {

        int csnroffst = (get_bits(gbc, 6) - 15) << 4;

        int snroffst = (csnroffst + get_bits(gbc, 4)) << 2;

        for (ch = 0; ch <= s->channels; ch++)

            s->snr_offset[ch] = snroffst;

    }



    /* transient pre-noise processing data */

    if (parse_transient_proc_info) {

        for (ch = 1; ch <= s->fbw_channels; ch++) {

            if (get_bits1(gbc)) { // channel in transient processing

                skip_bits(gbc, 10); // skip transient processing location

                skip_bits(gbc, 8);  // skip transient processing length

            }

        }

    }



    /* spectral extension attenuation data */

    for (ch = 1; ch <= s->fbw_channels; ch++) {

        if (parse_spx_atten_data && get_bits1(gbc)) {

            s->spx_atten_code[ch] = get_bits(gbc, 5);

        } else {

            s->spx_atten_code[ch] = -1;

        }

    }



    /* block start information */

    if (s->num_blocks > 1 && get_bits1(gbc)) {

        /* reference: Section E2.3.2.27

           nblkstrtbits = (numblks - 1) * (4 + ceiling(log2(words_per_frame)))

           The spec does not say what this data is or what it's used for.

           It is likely the offset of each block within the frame. */

        int block_start_bits = (s->num_blocks-1) * (4 + av_log2(s->frame_size-2));

        skip_bits_long(gbc, block_start_bits);

        avpriv_request_sample(s->avctx, "Block start info");

    }



    /* syntax state initialization */

    for (ch = 1; ch <= s->fbw_channels; ch++) {

        s->first_spx_coords[ch] = 1;

        s->first_cpl_coords[ch] = 1;

    }

    s->first_cpl_leak = 1;



    return 0;

}
