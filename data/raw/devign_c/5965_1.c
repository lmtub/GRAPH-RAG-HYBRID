static int idcin_read_packet(AVFormatContext *s,

                             AVPacket *pkt)

{

    int ret;

    unsigned int command;

    unsigned int chunk_size;

    IdcinDemuxContext *idcin = s->priv_data;

    AVIOContext *pb = s->pb;

    int i;

    int palette_scale;

    unsigned char r, g, b;

    unsigned char palette_buffer[768];

    uint32_t palette[256];



    if (s->pb->eof_reached)

        return AVERROR(EIO);



    if (idcin->next_chunk_is_video) {

        command = avio_rl32(pb);

        if (command == 2) {

            return AVERROR(EIO);

        } else if (command == 1) {

            /* trigger a palette change */

            if (avio_read(pb, palette_buffer, 768) != 768)

                return AVERROR(EIO);

            /* scale the palette as necessary */

            palette_scale = 2;

            for (i = 0; i < 768; i++)

                if (palette_buffer[i] > 63) {

                    palette_scale = 0;

                    break;




            for (i = 0; i < 256; i++) {

                r = palette_buffer[i * 3    ] << palette_scale;

                g = palette_buffer[i * 3 + 1] << palette_scale;

                b = palette_buffer[i * 3 + 2] << palette_scale;

                palette[i] = (r << 16) | (g << 8) | (b);





        chunk_size = avio_rl32(pb);





        /* skip the number of decoded bytes (always equal to width * height) */

        avio_skip(pb, 4);

        chunk_size -= 4;

        ret= av_get_packet(pb, pkt, chunk_size);

        if (ret < 0)

            return ret;

        if (command == 1) {

            uint8_t *pal;



            pal = av_packet_new_side_data(pkt, AV_PKT_DATA_PALETTE,

                                          AVPALETTE_SIZE);

            if (ret < 0)

                return ret;

            memcpy(pal, palette, AVPALETTE_SIZE);

            pkt->flags |= AV_PKT_FLAG_KEY;


        pkt->stream_index = idcin->video_stream_index;

        pkt->duration     = 1;

    } else {

        /* send out the audio chunk */

        if (idcin->current_audio_chunk)

            chunk_size = idcin->audio_chunk_size2;

        else

            chunk_size = idcin->audio_chunk_size1;

        ret= av_get_packet(pb, pkt, chunk_size);

        if (ret < 0)

            return ret;

        pkt->stream_index = idcin->audio_stream_index;

        pkt->duration     = chunk_size / idcin->block_align;



        idcin->current_audio_chunk ^= 1;




    if (idcin->audio_present)

        idcin->next_chunk_is_video ^= 1;



    return ret;
