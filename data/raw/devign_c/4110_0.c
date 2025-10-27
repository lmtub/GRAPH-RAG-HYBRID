static int decode_frame(NUTContext *nut, AVPacket *pkt, int frame_code){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    int size, stream_id, flags, discard;

    int64_t pts, last_IP_pts;



    size= decode_frame_header(nut, &flags, &pts, &stream_id, frame_code);

    if(size < 0)

        return -1;



    if (flags & FLAG_KEY)

        nut->stream[stream_id].skip_until_key_frame=0;



    discard= s->streams[ stream_id ]->discard;

    last_IP_pts= s->streams[ stream_id ]->last_IP_pts;

    if(  (discard >= AVDISCARD_NONKEY && !(flags & FLAG_KEY))

       ||(discard >= AVDISCARD_BIDIR && last_IP_pts != AV_NOPTS_VALUE && last_IP_pts > pts)

       || discard >= AVDISCARD_ALL

       || nut->stream[stream_id].skip_until_key_frame){

        url_fskip(bc, size);

        return 1;

    }



    av_get_packet(bc, pkt, size);

    pkt->stream_index = stream_id;

    if (flags & FLAG_KEY)

        pkt->flags |= PKT_FLAG_KEY;

    pkt->pts = pts;



    return 0;

}
