static int wav_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    int64_t size, av_uninit(data_size);

    int64_t sample_count=0;

    int rf64;

    unsigned int tag;

    AVIOContext *pb = s->pb;

    AVStream *st;

    WAVContext *wav = s->priv_data;

    int ret, got_fmt = 0;

    int64_t next_tag_ofs, data_ofs = -1;



    /* check RIFF header */

    tag = avio_rl32(pb);



    rf64 = tag == MKTAG('R', 'F', '6', '4');

    if (!rf64 && tag != MKTAG('R', 'I', 'F', 'F'))

        return -1;

    avio_rl32(pb); /* file size */

    tag = avio_rl32(pb);

    if (tag != MKTAG('W', 'A', 'V', 'E'))

        return -1;



    if (rf64) {

        if (avio_rl32(pb) != MKTAG('d', 's', '6', '4'))

            return -1;

        size = avio_rl32(pb);

        if (size < 16)

            return -1;

        avio_rl64(pb); /* RIFF size */

        data_size = avio_rl64(pb);

        sample_count = avio_rl64(pb);

        if (data_size < 0 || sample_count < 0) {

            av_log(s, AV_LOG_ERROR, "negative data_size and/or sample_count in "

                   "ds64: data_size = %"PRId64", sample_count = %"PRId64"\n",

                   data_size, sample_count);

            return AVERROR_INVALIDDATA;

        }

        avio_skip(pb, size - 24); /* skip rest of ds64 chunk */

    }



    for (;;) {

        size = next_tag(pb, &tag);

        next_tag_ofs = avio_tell(pb) + size;



        if (url_feof(pb))

            break;



        switch (tag) {

        case MKTAG('f', 'm', 't', ' '):

            /* only parse the first 'fmt ' tag found */

            if (!got_fmt && (ret = wav_parse_fmt_tag(s, size, &st) < 0)) {

                return ret;

            } else if (got_fmt)

                av_log(s, AV_LOG_WARNING, "found more than one 'fmt ' tag\n");



            got_fmt = 1;

            break;

        case MKTAG('d', 'a', 't', 'a'):

            if (!got_fmt) {

                av_log(s, AV_LOG_ERROR, "found no 'fmt ' tag before the 'data' tag\n");

                return AVERROR_INVALIDDATA;

            }



            if (rf64) {

                next_tag_ofs = wav->data_end = avio_tell(pb) + data_size;

            } else {

                data_size = size;

                next_tag_ofs = wav->data_end = size ? next_tag_ofs : INT64_MAX;

            }



            data_ofs = avio_tell(pb);



            /* don't look for footer metadata if we can't seek or if we don't

             * know where the data tag ends

             */

            if (!pb->seekable || (!rf64 && !size))

                goto break_loop;

            break;

        case MKTAG('f','a','c','t'):

            if(!sample_count)

                sample_count = avio_rl32(pb);

            break;

        case MKTAG('b','e','x','t'):

            if ((ret = wav_parse_bext_tag(s, size)) < 0)

                return ret;

            break;

        }



        /* seek to next tag unless we know that we'll run into EOF */

        if ((avio_size(pb) > 0 && next_tag_ofs >= avio_size(pb)) ||

            avio_seek(pb, next_tag_ofs, SEEK_SET) < 0) {

            break;

        }

    }

break_loop:

    if (data_ofs < 0) {

        av_log(s, AV_LOG_ERROR, "no 'data' tag found\n");

        return AVERROR_INVALIDDATA;

    }



    avio_seek(pb, data_ofs, SEEK_SET);



    if (!sample_count && st->codec->channels && av_get_bits_per_sample(st->codec->codec_id))

        sample_count = (data_size<<3) / (st->codec->channels * (uint64_t)av_get_bits_per_sample(st->codec->codec_id));

    if (sample_count)

        st->duration = sample_count;



    ff_metadata_conv_ctx(s, NULL, wav_metadata_conv);



    return 0;

}
