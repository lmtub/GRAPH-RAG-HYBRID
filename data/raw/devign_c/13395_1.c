static int parse_metadata(DBEContext *s)

{

    int i, ret, key = parse_key(s), mtd_size;



    if ((ret = convert_input(s, 1, key)) < 0)

        return ret;



    skip_bits(&s->gb, 4);

    mtd_size = get_bits(&s->gb, 10);

    if (!mtd_size) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid metadata size\n");

        return AVERROR_INVALIDDATA;

    }



    if ((ret = convert_input(s, mtd_size, key)) < 0)

        return ret;



    skip_bits(&s->gb, 14);

    s->prog_conf = get_bits(&s->gb, 6);

    if (s->prog_conf > MAX_PROG_CONF) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid program configuration\n");

        return AVERROR_INVALIDDATA;

    }



    s->nb_channels = nb_channels_tab[s->prog_conf];

    s->nb_programs = nb_programs_tab[s->prog_conf];



    s->fr_code      = get_bits(&s->gb, 4);

    s->fr_code_orig = get_bits(&s->gb, 4);

    if (!sample_rate_tab[s->fr_code] ||

        !sample_rate_tab[s->fr_code_orig]) {

        av_log(s->avctx, AV_LOG_ERROR, "Invalid frame rate code\n");

        return AVERROR_INVALIDDATA;

    }



    skip_bits_long(&s->gb, 88);

    for (i = 0; i < s->nb_channels; i++)

        s->ch_size[i] = get_bits(&s->gb, 10);

    s->mtd_ext_size = get_bits(&s->gb, 8);

    s->meter_size   = get_bits(&s->gb, 8);



    skip_bits_long(&s->gb, 10 * s->nb_programs);

    for (i = 0; i < s->nb_channels; i++) {

        s->rev_id[i]     = get_bits(&s->gb,  4);

        skip_bits1(&s->gb);

        s->begin_gain[i] = get_bits(&s->gb, 10);

        s->end_gain[i]   = get_bits(&s->gb, 10);

    }



    if (get_bits_left(&s->gb) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Read past end of metadata\n");

        return AVERROR_INVALIDDATA;

    }



    skip_input(s, mtd_size + 1);

    return 0;

}
