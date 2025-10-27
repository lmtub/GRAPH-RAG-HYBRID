static int asf_read_ext_content(AVFormatContext *s, const GUIDParseTable *g)

{

    ASFContext *asf  = s->priv_data;

    AVIOContext *pb  = s->pb;

    uint64_t size    = avio_rl64(pb);

    uint16_t nb_desc = avio_rl16(pb);

    int i, ret;



    for (i = 0; i < nb_desc; i++) {

        uint16_t name_len, type, val_len;

        uint8_t *name = NULL;



        name_len = avio_rl16(pb);

        if (!name_len)

            return AVERROR_INVALIDDATA;

        name = av_malloc(name_len);

        if (!name)

            return AVERROR(ENOMEM);

        avio_get_str16le(pb, name_len, name,

                         name_len);

        type    = avio_rl16(pb);

        val_len = avio_rl16(pb);



        if ((ret = process_metadata(s, name, name_len, val_len, type, &s->metadata)) < 0)

            return ret;

    }



    align_position(pb, asf->offset, size);

    return 0;

}
