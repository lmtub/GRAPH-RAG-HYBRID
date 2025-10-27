int avpriv_mpeg4audio_get_config(MPEG4AudioConfig *c, const uint8_t *buf,
                                 int bit_size, int sync_extension)
{
    GetBitContext gb;
    int specific_config_bitindex;
    init_get_bits(&gb, buf, bit_size);
    c->object_type = get_object_type(&gb);
    c->sample_rate = get_sample_rate(&gb, &c->sampling_index);
    c->chan_config = get_bits(&gb, 4);
    if (c->chan_config < FF_ARRAY_ELEMS(ff_mpeg4audio_channels))
        c->channels = ff_mpeg4audio_channels[c->chan_config];
    c->sbr = -1;
    c->ps  = -1;
    if (c->object_type == AOT_SBR || (c->object_type == AOT_PS &&
        // check for W6132 Annex YYYY draft MP3onMP4
        !(show_bits(&gb, 3) & 0x03 && !(show_bits(&gb, 9) & 0x3F)))) {
        if (c->object_type == AOT_PS)
            c->ps = 1;
        c->ext_object_type = AOT_SBR;
        c->sbr = 1;
        c->ext_sample_rate = get_sample_rate(&gb, &c->ext_sampling_index);
        c->object_type = get_object_type(&gb);
        if (c->object_type == AOT_ER_BSAC)
            c->ext_chan_config = get_bits(&gb, 4);
    } else {
        c->ext_object_type = AOT_NULL;
        c->ext_sample_rate = 0;
    }
    specific_config_bitindex = get_bits_count(&gb);
    if (c->object_type == AOT_ALS) {
        skip_bits(&gb, 5);
        if (show_bits_long(&gb, 24) != MKBETAG('\0','A','L','S'))
            skip_bits_long(&gb, 24);
        specific_config_bitindex = get_bits_count(&gb);
        if (parse_config_ALS(&gb, c))
            return -1;
    }
    if (c->ext_object_type != AOT_SBR && sync_extension) {
        while (get_bits_left(&gb) > 15) {
            if (show_bits(&gb, 11) == 0x2b7) { // sync extension
                get_bits(&gb, 11);
                c->ext_object_type = get_object_type(&gb);
                if (c->ext_object_type == AOT_SBR && (c->sbr = get_bits1(&gb)) == 1)
                    c->ext_sample_rate = get_sample_rate(&gb, &c->ext_sampling_index);
                if (get_bits_left(&gb) > 11 && get_bits(&gb, 11) == 0x548)
                    c->ps = get_bits1(&gb);
                break;
            } else
                get_bits1(&gb); // skip 1 bit
        }
    }
    //PS requires SBR
    if (!c->sbr)
        c->ps = 0;
    //Limit implicit PS to the HE-AACv2 Profile
    if ((c->ps == -1 && c->object_type != AOT_AAC_LC) || c->channels & ~0x01)
        c->ps = 0;
    return specific_config_bitindex;
}