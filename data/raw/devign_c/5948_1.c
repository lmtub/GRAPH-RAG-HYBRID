static int asf_read_metadata_obj(AVFormatContext *s, const GUIDParseTable *g)

{

    ASFContext *asf   = s->priv_data;

    AVIOContext *pb   = s->pb;

    uint64_t size     = avio_rl64(pb);

    uint16_t nb_recs  = avio_rl16(pb); // number of records in the Description Records list

    int i, ret;



    for (i = 0; i < nb_recs; i++) {

        uint16_t name_len, buflen, type, val_len, st_num;

        uint8_t *name = NULL;



        avio_skip(pb, 2); // skip reserved field

        st_num   = avio_rl16(pb);

        name_len = avio_rl16(pb);

        buflen   = 2 * name_len + 1;

        if (!name_len)

            break;

        type     = avio_rl16(pb);

        val_len  = avio_rl32(pb);

        name     = av_malloc(name_len);

        if (!name)

            return AVERROR(ENOMEM);

        avio_get_str16le(pb, name_len, name,

                         buflen);



        if (!strcmp(name, "AspectRatioX") || !strcmp(name, "AspectRatioY")) {

            asf_store_aspect_ratio(s, st_num, name);

        } else {

            if (st_num < ASF_MAX_STREAMS) {

                if ((ret = process_metadata(s, name, name_len, val_len, type,

                                            &asf->asf_sd[st_num].asf_met)) < 0)

                    break;

            } else

                av_freep(&name);

        }

    }



    align_position(pb, asf->offset, size);

    return 0;

}
