static int w64_read_header(AVFormatContext *s)

{

    int64_t size, data_ofs = 0;

    AVIOContext *pb      = s->pb;

    WAVDemuxContext *wav = s->priv_data;

    AVStream *st;

    uint8_t guid[16];

    int ret;



    avio_read(pb, guid, 16);

    if (memcmp(guid, ff_w64_guid_riff, 16))




    /* riff + wave + fmt + sizes */

    if (avio_rl64(pb) < 16 + 8 + 16 + 8 + 16 + 8)




    avio_read(pb, guid, 16);

    if (memcmp(guid, ff_w64_guid_wave, 16)) {

        av_log(s, AV_LOG_ERROR, "could not find wave guid\n");


    }



    wav->w64 = 1;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    while (!avio_feof(pb)) {

        if (avio_read(pb, guid, 16) != 16)

            break;

        size = avio_rl64(pb);

        if (size <= 24 || INT64_MAX - size < avio_tell(pb))




        if (!memcmp(guid, ff_w64_guid_fmt, 16)) {

            /* subtract chunk header size - normal wav file doesn't count it */

            ret = ff_get_wav_header(s, pb, st->codecpar, size - 24, 0);

            if (ret < 0)

                return ret;

            avio_skip(pb, FFALIGN(size, INT64_C(8)) - size);



            avpriv_set_pts_info(st, 64, 1, st->codecpar->sample_rate);

        } else if (!memcmp(guid, ff_w64_guid_fact, 16)) {

            int64_t samples;



            samples = avio_rl64(pb);

            if (samples > 0)

                st->duration = samples;

        } else if (!memcmp(guid, ff_w64_guid_data, 16)) {

            wav->data_end = avio_tell(pb) + size - 24;



            data_ofs = avio_tell(pb);

            if (!(pb->seekable & AVIO_SEEKABLE_NORMAL))

                break;



            avio_skip(pb, size - 24);

        } else if (!memcmp(guid, ff_w64_guid_summarylist, 16)) {

            int64_t start, end, cur;

            uint32_t count, chunk_size, i;



            start = avio_tell(pb);

            end = start + FFALIGN(size, INT64_C(8)) - 24;

            count = avio_rl32(pb);



            for (i = 0; i < count; i++) {

                char chunk_key[5], *value;



                if (avio_feof(pb) || (cur = avio_tell(pb)) < 0 || cur > end - 8 /* = tag + size */)

                    break;



                chunk_key[4] = 0;

                avio_read(pb, chunk_key, 4);

                chunk_size = avio_rl32(pb);





                value = av_mallocz(chunk_size + 1);

                if (!value)

                    return AVERROR(ENOMEM);



                ret = avio_get_str16le(pb, chunk_size, value, chunk_size);

                avio_skip(pb, chunk_size - ret);



                av_dict_set(&s->metadata, chunk_key, value, AV_DICT_DONT_STRDUP_VAL);

            }



            avio_skip(pb, end - avio_tell(pb));

        } else {

            av_log(s, AV_LOG_DEBUG, "unknown guid: "FF_PRI_GUID"\n", FF_ARG_GUID(guid));

            avio_skip(pb, FFALIGN(size, INT64_C(8)) - 24);

        }

    }



    if (!data_ofs)

        return AVERROR_EOF;



    ff_metadata_conv_ctx(s, NULL, wav_metadata_conv);

    ff_metadata_conv_ctx(s, NULL, ff_riff_info_conv);



    handle_stream_probing(st);

    st->need_parsing = AVSTREAM_PARSE_FULL_RAW;



    avio_seek(pb, data_ofs, SEEK_SET);



    set_spdif(s, wav);



    return 0;

}