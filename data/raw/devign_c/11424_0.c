static int mpeg_mux_write_packet(AVFormatContext *ctx, int stream_index,

                                 const uint8_t *buf, int size, int64_t pts)

{

    MpegMuxContext *s = ctx->priv_data;

    AVStream *st = ctx->streams[stream_index];

    StreamInfo *stream = st->priv_data;

    int len;

    

    while (size > 0) {

        /* set pts */

        if (stream->start_pts == -1) {

            stream->start_pts = pts;

        }

        len = s->packet_data_max_size - stream->buffer_ptr;

        if (len > size)

            len = size;

        memcpy(stream->buffer + stream->buffer_ptr, buf, len);

        stream->buffer_ptr += len;

        buf += len;

        size -= len;

        while (stream->buffer_ptr >= s->packet_data_max_size) {

            /* output the packet */

            if (stream->start_pts == -1)

                stream->start_pts = pts;

            flush_packet(ctx, stream_index, 0);

        }

    }

    return 0;

}
