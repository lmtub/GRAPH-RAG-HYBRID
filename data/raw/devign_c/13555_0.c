int ff_dca_core_parse(DCACoreDecoder *s, uint8_t *data, int size)

{

    int ret;



    s->ext_audio_mask = 0;

    s->xch_pos = s->xxch_pos = s->x96_pos = 0;



    if ((ret = init_get_bits8(&s->gb, data, size)) < 0)

        return ret;

    s->gb_in = s->gb;



    if ((ret = parse_frame_header(s)) < 0)

        return ret;

    if ((ret = alloc_sample_buffer(s)) < 0)

        return ret;

    if ((ret = parse_frame_data(s, HEADER_CORE, 0)) < 0)

        return ret;

    if ((ret = parse_optional_info(s)) < 0)

        return ret;



    // Workaround for DTS in WAV

    if (s->frame_size > size && s->frame_size < size + 4)

        s->frame_size = size;



    if (ff_dca_seek_bits(&s->gb, s->frame_size * 8)) {

        av_log(s->avctx, AV_LOG_ERROR, "Read past end of core frame\n");

        if (s->avctx->err_recognition & AV_EF_EXPLODE)

            return AVERROR_INVALIDDATA;

    }



    return 0;

}
