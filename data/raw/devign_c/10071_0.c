static void flush_packet(AVFormatContext *ctx, int stream_index, int last_pkt)

{

    MpegMuxContext *s = ctx->priv_data;

    StreamInfo *stream = ctx->streams[stream_index]->priv_data;

    uint8_t *buf_ptr;

    int size, payload_size, startcode, id, len, stuffing_size, i, header_len;

    int64_t timestamp;

    uint8_t buffer[128];

    int last = last_pkt ? 4 : 0;

    

    id = stream->id;

    timestamp = stream->start_pts;



#if 0

    printf("packet ID=%2x PTS=%0.3f\n", 

           id, timestamp / 90000.0);

#endif



    buf_ptr = buffer;

    if (((s->packet_number % s->pack_header_freq) == 0)) {

        /* output pack and systems header if needed */

        size = put_pack_header(ctx, buf_ptr, timestamp);

        buf_ptr += size;

        if ((s->packet_number % s->system_header_freq) == 0) {

            size = put_system_header(ctx, buf_ptr);

            buf_ptr += size;

        }

    }

    size = buf_ptr - buffer;

    put_buffer(&ctx->pb, buffer, size);



    /* packet header */

    if (s->is_mpeg2) {

        header_len = 8;

    } else {

        header_len = 5;

    }

    payload_size = s->packet_size - (size + 6 + header_len + last);

    if (id < 0xc0) {

        startcode = PRIVATE_STREAM_1;

        payload_size -= 4;

    } else {

        startcode = 0x100 + id;

    }

    stuffing_size = payload_size - stream->buffer_ptr;

    if (stuffing_size < 0)

        stuffing_size = 0;



    put_be32(&ctx->pb, startcode);



    put_be16(&ctx->pb, payload_size + header_len);

    /* stuffing */

    for(i=0;i<stuffing_size;i++)

        put_byte(&ctx->pb, 0xff);



    if (s->is_mpeg2) {

        put_byte(&ctx->pb, 0x80); /* mpeg2 id */

        put_byte(&ctx->pb, 0x80); /* flags */

        put_byte(&ctx->pb, 0x05); /* header len (only pts is included) */

    }

    put_byte(&ctx->pb, 

             (0x02 << 4) | 

             (((timestamp >> 30) & 0x07) << 1) | 

             1);

    put_be16(&ctx->pb, (uint16_t)((((timestamp >> 15) & 0x7fff) << 1) | 1));

    put_be16(&ctx->pb, (uint16_t)((((timestamp) & 0x7fff) << 1) | 1));



    if (startcode == PRIVATE_STREAM_1) {

        put_byte(&ctx->pb, id);

        if (id >= 0x80 && id <= 0xbf) {

            /* XXX: need to check AC3 spec */

            put_byte(&ctx->pb, 1);

            put_byte(&ctx->pb, 0);

            put_byte(&ctx->pb, 2);

        }

    }



    if (last_pkt) {

        put_be32(&ctx->pb, ISO_11172_END_CODE);

    }

    /* output data */

    put_buffer(&ctx->pb, stream->buffer, payload_size - stuffing_size);

    put_flush_packet(&ctx->pb);

    

    /* preserve remaining data */

    len = stream->buffer_ptr - payload_size;

    if (len < 0) 

        len = 0;

    memmove(stream->buffer, stream->buffer + stream->buffer_ptr - len, len);

    stream->buffer_ptr = len;



    s->packet_number++;

    stream->packet_number++;

    stream->start_pts = -1;

}
