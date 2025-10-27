static int dca_parse_params(DCAParseContext *pc1, const uint8_t *buf,

                            int buf_size, int *duration, int *sample_rate,

                            int *profile)

{

    DCAExssAsset *asset = &pc1->exss.assets[0];

    GetBitContext gb;

    DCACoreFrameHeader h;

    uint8_t hdr[DCA_CORE_FRAME_HEADER_SIZE + AV_INPUT_BUFFER_PADDING_SIZE] = { 0 };

    int ret, frame_size;



    if (buf_size < DCA_CORE_FRAME_HEADER_SIZE)

        return AVERROR_INVALIDDATA;



    if (AV_RB32(buf) == DCA_SYNCWORD_SUBSTREAM) {

        if ((ret = ff_dca_exss_parse(&pc1->exss, buf, buf_size)) < 0)

            return ret;



        if (asset->extension_mask & DCA_EXSS_LBR) {

            if ((ret = init_get_bits8(&gb, buf + asset->lbr_offset, asset->lbr_size)) < 0)

                return ret;



            if (get_bits_long(&gb, 32) != DCA_SYNCWORD_LBR)

                return AVERROR_INVALIDDATA;



            switch (get_bits(&gb, 8)) {

            case DCA_LBR_HEADER_DECODER_INIT:

                pc1->sr_code = get_bits(&gb, 8);

            case DCA_LBR_HEADER_SYNC_ONLY:

                break;

            default:

                return AVERROR_INVALIDDATA;

            }



            if (pc1->sr_code >= FF_ARRAY_ELEMS(ff_dca_sampling_freqs))

                return AVERROR_INVALIDDATA;



            *sample_rate = ff_dca_sampling_freqs[pc1->sr_code];

            *duration = 1024 << ff_dca_freq_ranges[pc1->sr_code];

            *profile = FF_PROFILE_DTS_EXPRESS;

            return 0;

        }



        if (asset->extension_mask & DCA_EXSS_XLL) {

            int nsamples_log2;



            if ((ret = init_get_bits8(&gb, buf + asset->xll_offset, asset->xll_size)) < 0)

                return ret;



            if (get_bits_long(&gb, 32) != DCA_SYNCWORD_XLL)

                return AVERROR_INVALIDDATA;



            if (get_bits(&gb, 4))

                return AVERROR_INVALIDDATA;



            skip_bits(&gb, 8);

            skip_bits_long(&gb, get_bits(&gb, 5) + 1);

            skip_bits(&gb, 4);

            nsamples_log2 = get_bits(&gb, 4) + get_bits(&gb, 4);

            if (nsamples_log2 > 24)

                return AVERROR_INVALIDDATA;



            *sample_rate = asset->max_sample_rate;

            *duration = (1 + (*sample_rate > 96000)) << nsamples_log2;

            *profile = FF_PROFILE_DTS_HD_MA;

            return 0;

        }



        return AVERROR_INVALIDDATA;

    }



    if ((ret = avpriv_dca_convert_bitstream(buf, DCA_CORE_FRAME_HEADER_SIZE,

                                            hdr, DCA_CORE_FRAME_HEADER_SIZE)) < 0)

        return ret;

    if ((ret = init_get_bits8(&gb, hdr, ret)) < 0)

        return ret;

    if (avpriv_dca_parse_core_frame_header(&gb, &h) < 0)

        return AVERROR_INVALIDDATA;



    *duration = h.npcmblocks * DCA_PCMBLOCK_SAMPLES;

    *sample_rate = avpriv_dca_sample_rates[h.sr_code];

    if (*profile != FF_PROFILE_UNKNOWN)

        return 0;



    *profile = FF_PROFILE_DTS;

    if (h.ext_audio_present) {

        switch (h.ext_audio_type) {

        case DCA_EXT_AUDIO_XCH:

        case DCA_EXT_AUDIO_XXCH:

            *profile = FF_PROFILE_DTS_ES;

            break;

        case DCA_EXT_AUDIO_X96:

            *profile = FF_PROFILE_DTS_96_24;

            break;

        }

    }



    frame_size = FFALIGN(h.frame_size, 4);

    if (buf_size - 4 < frame_size)

        return 0;



    buf      += frame_size;

    buf_size -= frame_size;

    if (AV_RB32(buf) != DCA_SYNCWORD_SUBSTREAM)

        return 0;

    if (ff_dca_exss_parse(&pc1->exss, buf, buf_size) < 0)

        return 0;



    if (asset->extension_mask & DCA_EXSS_XLL)

        *profile = FF_PROFILE_DTS_HD_MA;

    else if (asset->extension_mask & (DCA_EXSS_XBR | DCA_EXSS_XXCH | DCA_EXSS_X96))

        *profile = FF_PROFILE_DTS_HD_HRA;



    return 0;

}
