static void ffm_seek1(AVFormatContext *s, int64_t pos1)

{

    FFMContext *ffm = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t pos;



    pos = FFMIN(pos1, ffm->file_size - FFM_PACKET_SIZE);

    pos = FFMAX(pos, FFM_PACKET_SIZE);

    av_dlog(s, "seek to %"PRIx64" -> %"PRIx64"\n", pos1, pos);

    avio_seek(pb, pos, SEEK_SET);

}
