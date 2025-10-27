static int flac_write_trailer(struct AVFormatContext *s)

{

    ByteIOContext *pb = s->pb;

    uint8_t *streaminfo = s->streams[0]->codec->extradata;

    int len = s->streams[0]->codec->extradata_size;

    int64_t file_size;



    if (streaminfo && len > 0 && !url_is_streamed(s->pb)) {

        file_size = url_ftell(pb);

        url_fseek(pb, 8, SEEK_SET);

        put_buffer(pb, streaminfo, len);

        url_fseek(pb, file_size, SEEK_SET);

        put_flush_packet(pb);

    }

    return 0;

}
