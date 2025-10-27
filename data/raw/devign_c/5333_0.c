static int decode_ics_info(AACContext *ac, IndividualChannelStream *ics,

                           GetBitContext *gb)

{

    if (get_bits1(gb)) {

        av_log(ac->avctx, AV_LOG_ERROR, "Reserved bit set.\n");

        return AVERROR_INVALIDDATA;

    }

    ics->window_sequence[1] = ics->window_sequence[0];

    ics->window_sequence[0] = get_bits(gb, 2);

    ics->use_kb_window[1]   = ics->use_kb_window[0];

    ics->use_kb_window[0]   = get_bits1(gb);

    ics->num_window_groups  = 1;

    ics->group_len[0]       = 1;

    if (ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE) {

        int i;

        ics->max_sfb = get_bits(gb, 4);

        for (i = 0; i < 7; i++) {

            if (get_bits1(gb)) {

                ics->group_len[ics->num_window_groups - 1]++;

            } else {

                ics->num_window_groups++;

                ics->group_len[ics->num_window_groups - 1] = 1;

            }

        }

        ics->num_windows       = 8;

        ics->swb_offset        =    ff_swb_offset_128[ac->m4ac.sampling_index];

        ics->num_swb           =   ff_aac_num_swb_128[ac->m4ac.sampling_index];

        ics->tns_max_bands     = ff_tns_max_bands_128[ac->m4ac.sampling_index];

        ics->predictor_present = 0;

    } else {

        ics->max_sfb               = get_bits(gb, 6);

        ics->num_windows           = 1;

        ics->swb_offset            =    ff_swb_offset_1024[ac->m4ac.sampling_index];

        ics->num_swb               =   ff_aac_num_swb_1024[ac->m4ac.sampling_index];

        ics->tns_max_bands         = ff_tns_max_bands_1024[ac->m4ac.sampling_index];

        ics->predictor_present     = get_bits1(gb);

        ics->predictor_reset_group = 0;

        if (ics->predictor_present) {

            if (ac->m4ac.object_type == AOT_AAC_MAIN) {

                if (decode_prediction(ac, ics, gb)) {

                    return AVERROR_INVALIDDATA;

                }

            } else if (ac->m4ac.object_type == AOT_AAC_LC) {

                av_log(ac->avctx, AV_LOG_ERROR, "Prediction is not allowed in AAC-LC.\n");

                return AVERROR_INVALIDDATA;

            } else {

                if ((ics->ltp.present = get_bits(gb, 1)))

                    decode_ltp(ac, &ics->ltp, gb, ics->max_sfb);

            }

        }

    }



    if (ics->max_sfb > ics->num_swb) {

        av_log(ac->avctx, AV_LOG_ERROR,

               "Number of scalefactor bands in group (%d) exceeds limit (%d).\n",

               ics->max_sfb, ics->num_swb);

        return AVERROR_INVALIDDATA;

    }



    return 0;

}
