static int rm_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    if (s->streams[pkt->stream_index]->codec.codec_type == 

        CODEC_TYPE_AUDIO)

        return rm_write_audio(s, pkt->data, pkt->size);

    else

        return rm_write_video(s, pkt->data, pkt->size);

}
