ogg_read_header (AVFormatContext * s, AVFormatParameters * ap)
{
    struct ogg *ogg = s->priv_data;
    ogg->curidx = -1;
    //linear headers seek from start
    if (ogg_get_headers (s) < 0){
        return -1;
    }
    //linear granulepos seek from end
    ogg_get_length (s);
    //fill the extradata in the per codec callbacks
    return 0;
}