int avpriv_dca_parse_core_frame_header(GetBitContext *gb, DCACoreFrameHeader *h)

{

    if (get_bits_long(gb, 32) != DCA_SYNCWORD_CORE_BE)

        return DCA_PARSE_ERROR_SYNC_WORD;



    h->normal_frame = get_bits1(gb);

    h->deficit_samples = get_bits(gb, 5) + 1;

    if (h->deficit_samples != DCA_PCMBLOCK_SAMPLES)

        return DCA_PARSE_ERROR_DEFICIT_SAMPLES;



    h->crc_present = get_bits1(gb);

    h->npcmblocks = get_bits(gb, 7) + 1;

    if (h->npcmblocks & (DCA_SUBBAND_SAMPLES - 1))

        return DCA_PARSE_ERROR_PCM_BLOCKS;



    h->frame_size = get_bits(gb, 14) + 1;

    if (h->frame_size < 96)

        return DCA_PARSE_ERROR_FRAME_SIZE;



    h->audio_mode = get_bits(gb, 6);

    if (h->audio_mode >= DCA_AMODE_COUNT)

        return DCA_PARSE_ERROR_AMODE;



    h->sr_code = get_bits(gb, 4);

    if (!avpriv_dca_sample_rates[h->sr_code])

        return DCA_PARSE_ERROR_SAMPLE_RATE;



    h->br_code = get_bits(gb, 5);

    if (get_bits1(gb))

        return DCA_PARSE_ERROR_RESERVED_BIT;



    h->drc_present = get_bits1(gb);

    h->ts_present = get_bits1(gb);

    h->aux_present = get_bits1(gb);

    h->hdcd_master = get_bits1(gb);

    h->ext_audio_type = get_bits(gb, 3);

    h->ext_audio_present = get_bits1(gb);

    h->sync_ssf = get_bits1(gb);

    h->lfe_present = get_bits(gb, 2);

    if (h->lfe_present == DCA_LFE_FLAG_INVALID)

        return DCA_PARSE_ERROR_LFE_FLAG;



    h->predictor_history = get_bits1(gb);

    if (h->crc_present)

        skip_bits(gb, 16);

    h->filter_perfect = get_bits1(gb);

    h->encoder_rev = get_bits(gb, 4);

    h->copy_hist = get_bits(gb, 2);

    h->pcmr_code = get_bits(gb, 3);

    if (!ff_dca_bits_per_sample[h->pcmr_code])

        return DCA_PARSE_ERROR_PCM_RES;



    h->sumdiff_front = get_bits1(gb);

    h->sumdiff_surround = get_bits1(gb);

    h->dn_code = get_bits(gb, 4);

    return 0;

}
