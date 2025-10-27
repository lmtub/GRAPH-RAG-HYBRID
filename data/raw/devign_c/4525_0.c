int ff_read_riff_info(AVFormatContext *s, int64_t size)

{

    int64_t start, end, cur;

    AVIOContext *pb = s->pb;



    start = avio_tell(pb);

    end = start + size;



    while ((cur = avio_tell(pb)) >= 0 && cur <= end - 8 /* = tag + size */) {

        uint32_t chunk_code;

        int64_t chunk_size;

        char key[5] = {0};

        char *value;



        chunk_code = avio_rl32(pb);

        chunk_size = avio_rl32(pb);



        if (chunk_size > end || end - chunk_size < cur || chunk_size == UINT_MAX) {

            avio_seek(pb, -9, SEEK_CUR);

            chunk_code = avio_rl32(pb);

            chunk_size = avio_rl32(pb);

            if (chunk_size > end || end - chunk_size < cur || chunk_size == UINT_MAX) {

                av_log(s, AV_LOG_WARNING, "too big INFO subchunk\n");

                return AVERROR_INVALIDDATA;

            }

        }



        chunk_size += (chunk_size & 1);



        if (!chunk_code) {

            if (chunk_size)

                avio_skip(pb, chunk_size);

            continue;

        }



        value = av_mallocz(chunk_size + 1);

        if (!value) {

            av_log(s, AV_LOG_ERROR, "out of memory, unable to read INFO tag\n");

            return AVERROR(ENOMEM);

        }



        AV_WL32(key, chunk_code);



        if (avio_read(pb, value, chunk_size) != chunk_size) {

            av_log(s, AV_LOG_WARNING, "premature end of file while reading INFO tag\n");

        }



        av_dict_set(&s->metadata, key, value, AV_DICT_DONT_STRDUP_VAL);

    }



    return 0;

}
