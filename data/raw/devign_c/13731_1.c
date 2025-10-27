static int vmd_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    VmdDemuxContext *vmd = (VmdDemuxContext *)s->priv_data;

    ByteIOContext *pb = &s->pb;

    int ret = 0;

    vmd_frame_t *frame;



    if (vmd->current_frame >= vmd->frame_count)

        return -EIO;



    frame = &vmd->frame_table[vmd->current_frame];

    /* position the stream (will probably be there already) */

    url_fseek(pb, frame->frame_offset, SEEK_SET);



    if (av_new_packet(pkt, frame->frame_size + BYTES_PER_FRAME_RECORD))

        return AVERROR_NOMEM;

    memcpy(pkt->data, frame->frame_record, BYTES_PER_FRAME_RECORD);

    ret = get_buffer(pb, pkt->data + BYTES_PER_FRAME_RECORD, 

        frame->frame_size);



    if (ret != frame->frame_size)

        ret = -EIO;

    pkt->stream_index = frame->stream_index;

    pkt->pts = frame->pts;



    vmd->current_frame++;



    return ret;

}
