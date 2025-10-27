int avpriv_ac3_parse_header(GetBitContext *gbc, AC3HeaderInfo *hdr)

{

    int frame_size_code;



    memset(hdr, 0, sizeof(*hdr));



    hdr->sync_word = get_bits(gbc, 16);

    if(hdr->sync_word != 0x0B77)

        return AAC_AC3_PARSE_ERROR_SYNC;



    /* read ahead to bsid to distinguish between AC-3 and E-AC-3 */

    hdr->bitstream_id = show_bits_long(gbc, 29) & 0x1F;

    if(hdr->bitstream_id > 16)

        return AAC_AC3_PARSE_ERROR_BSID;



    hdr->num_blocks = 6;



    /* set default mix levels */

    hdr->center_mix_level   = 1;  // -4.5dB

    hdr->surround_mix_level = 1;  // -6.0dB



    if(hdr->bitstream_id <= 10) {

        /* Normal AC-3 */

        hdr->crc1 = get_bits(gbc, 16);

        hdr->sr_code = get_bits(gbc, 2);

        if(hdr->sr_code == 3)

            return AAC_AC3_PARSE_ERROR_SAMPLE_RATE;



        frame_size_code = get_bits(gbc, 6);

        if(frame_size_code > 37)

            return AAC_AC3_PARSE_ERROR_FRAME_SIZE;



        skip_bits(gbc, 5); // skip bsid, already got it



        hdr->bitstream_mode = get_bits(gbc, 3);

        hdr->channel_mode = get_bits(gbc, 3);



        if(hdr->channel_mode == AC3_CHMODE_STEREO) {

            skip_bits(gbc, 2); // skip dsurmod

        } else {

            if((hdr->channel_mode & 1) && hdr->channel_mode != AC3_CHMODE_MONO)

                hdr->center_mix_level = get_bits(gbc, 2);

            if(hdr->channel_mode & 4)

                hdr->surround_mix_level = get_bits(gbc, 2);

        }

        hdr->lfe_on = get_bits1(gbc);



        hdr->sr_shift = FFMAX(hdr->bitstream_id, 8) - 8;

        hdr->sample_rate = ff_ac3_sample_rate_tab[hdr->sr_code] >> hdr->sr_shift;

        hdr->bit_rate = (ff_ac3_bitrate_tab[frame_size_code>>1] * 1000) >> hdr->sr_shift;

        hdr->channels = ff_ac3_channels_tab[hdr->channel_mode] + hdr->lfe_on;

        hdr->frame_size = ff_ac3_frame_size_tab[frame_size_code][hdr->sr_code] * 2;

        hdr->frame_type = EAC3_FRAME_TYPE_AC3_CONVERT; //EAC3_FRAME_TYPE_INDEPENDENT;

        hdr->substreamid = 0;

    } else {

        /* Enhanced AC-3 */

        hdr->crc1 = 0;

        hdr->frame_type = get_bits(gbc, 2);

        if(hdr->frame_type == EAC3_FRAME_TYPE_RESERVED)

            return AAC_AC3_PARSE_ERROR_FRAME_TYPE;



        hdr->substreamid = get_bits(gbc, 3);



        hdr->frame_size = (get_bits(gbc, 11) + 1) << 1;

        if(hdr->frame_size < AC3_HEADER_SIZE)

            return AAC_AC3_PARSE_ERROR_FRAME_SIZE;



        hdr->sr_code = get_bits(gbc, 2);

        if (hdr->sr_code == 3) {

            int sr_code2 = get_bits(gbc, 2);

            if(sr_code2 == 3)

                return AAC_AC3_PARSE_ERROR_SAMPLE_RATE;

            hdr->sample_rate = ff_ac3_sample_rate_tab[sr_code2] / 2;

            hdr->sr_shift = 1;

        } else {

            hdr->num_blocks = eac3_blocks[get_bits(gbc, 2)];

            hdr->sample_rate = ff_ac3_sample_rate_tab[hdr->sr_code];

            hdr->sr_shift = 0;

        }



        hdr->channel_mode = get_bits(gbc, 3);

        hdr->lfe_on = get_bits1(gbc);



        hdr->bit_rate = (uint32_t)(8.0 * hdr->frame_size * hdr->sample_rate /

                        (hdr->num_blocks * 256.0));

        hdr->channels = ff_ac3_channels_tab[hdr->channel_mode] + hdr->lfe_on;

    }

    hdr->channel_layout = ff_ac3_channel_layout_tab[hdr->channel_mode];

    if (hdr->lfe_on)

        hdr->channel_layout |= AV_CH_LOW_FREQUENCY;



    return 0;

}
