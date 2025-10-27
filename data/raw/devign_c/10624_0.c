static int img_read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    VideoDemuxData *s1 = s->priv_data;



    if (timestamp < 0 || timestamp > s1->img_last - s1->img_first)

        return -1;

    s1->img_number = timestamp + s1->img_first;

    return 0;

}
