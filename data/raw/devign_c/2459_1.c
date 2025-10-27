static void flush_packet(AVFormatContext *ctx, int stream_index, 

                         int64_t pts, int64_t dts, int64_t scr)

{

    MpegMuxContext *s = ctx->priv_data;

    StreamInfo *stream = ctx->streams[stream_index]->priv_data;

    uint8_t *buf_ptr;

    int size, payload_size, startcode, id, len, stuffing_size, i, header_len;

    uint8_t buffer[128];

    

    id = stream->id;

    

#if 0

    printf("packet ID=%2x PTS=%0.3f\n", 

           id, pts / 90000.0);

#endif



    buf_ptr = buffer;

    if (((s->packet_number % s->pack_header_freq) == 0)) {

        /* output pack and systems header if needed */

        size = put_pack_header(ctx, buf_ptr, scr);

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

        header_len = 3;

    } else {

        header_len = 0;

    }

    if (pts != AV_NOPTS_VALUE) {

        if (dts != AV_NOPTS_VALUE)

            header_len += 5 + 5;

        else

            header_len += 5;

    } else {

        if (!s->is_mpeg2)

            header_len++;

    }



    payload_size = s->packet_size - (size + 6 + header_len);

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



        if (pts != AV_NOPTS_VALUE) {

            if (dts != AV_NOPTS_VALUE) {

                put_byte(&ctx->pb, 0xc0); /* flags */

                put_byte(&ctx->pb, header_len - 3);

                put_timestamp(&ctx->pb, 0x03, pts);

                put_timestamp(&ctx->pb, 0x01, dts);

            } else {

                put_byte(&ctx->pb, 0x80); /* flags */

                put_byte(&ctx->pb, header_len - 3);

                put_timestamp(&ctx->pb, 0x02, pts);

            }

        } else {

            put_byte(&ctx->pb, 0x00); /* flags */

            put_byte(&ctx->pb, header_len - 3);

        }

    } else {

        if (pts != AV_NOPTS_VALUE) {

            if (dts != AV_NOPTS_VALUE) {

                put_timestamp(&ctx->pb, 0x03, pts);

                put_timestamp(&ctx->pb, 0x01, dts);

            } else {

                put_timestamp(&ctx->pb, 0x02, pts);

            }

        } else {

            put_byte(&ctx->pb, 0x0f);

        }

    }



    if (startcode == PRIVATE_STREAM_1) {

        put_byte(&ctx->pb, id);

        if (id >= 0x80 && id <= 0xbf) {

            /* XXX: need to check AC3 spec */

            put_byte(&ctx->pb, 1);

            put_byte(&ctx->pb, 0);

            put_byte(&ctx->pb, 2);

        }

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

}
