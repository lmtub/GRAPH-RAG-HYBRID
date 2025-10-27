static int yop_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    YopDecContext *yop = s->priv_data;

    AVIOContext *pb  = s->pb;



    int ret;

    int actual_video_data_size = yop->frame_size -

                                 yop->audio_block_length - yop->palette_size;



    yop->video_packet.stream_index = 1;



    if (yop->video_packet.data) {

        *pkt                   =  yop->video_packet;

        yop->video_packet.data =  NULL;

        yop->video_packet.size =  0;

        pkt->data[0]           =  yop->odd_frame;

        pkt->flags             |= AV_PKT_FLAG_KEY;

        yop->odd_frame         ^= 1;

        return pkt->size;

    }

    ret = av_new_packet(&yop->video_packet,

                        yop->frame_size - yop->audio_block_length);

    if (ret < 0)

        return ret;



    yop->video_packet.pos = avio_tell(pb);



    ret = avio_read(pb, yop->video_packet.data, yop->palette_size);

    if (ret < 0) {

        goto err_out;

    }else if (ret < yop->palette_size) {

        ret = AVERROR_EOF;

        goto err_out;

    }



    ret = av_get_packet(pb, pkt, 920);

    if (ret < 0)

        goto err_out;



    // Set position to the start of the frame

    pkt->pos = yop->video_packet.pos;



    avio_skip(pb, yop->audio_block_length - ret);



    ret = avio_read(pb, yop->video_packet.data + yop->palette_size,

                     actual_video_data_size);

    if (ret < 0)

        goto err_out;

    else if (ret < actual_video_data_size)

        av_shrink_packet(&yop->video_packet, yop->palette_size + ret);



    // Arbitrarily return the audio data first

    return yop->audio_block_length;



err_out:

    av_free_packet(&yop->video_packet);

    return ret;

}
