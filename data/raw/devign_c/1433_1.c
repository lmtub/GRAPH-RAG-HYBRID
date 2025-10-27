static int dss_read_metadata_date(AVFormatContext *s, unsigned int offset,

                                  const char *key)

{

    AVIOContext *pb = s->pb;

    char datetime[64], string[DSS_TIME_SIZE + 1] = { 0 };

    int y, month, d, h, minute, sec;

    int ret;



    avio_seek(pb, offset, SEEK_SET);



    ret = avio_read(s->pb, string, DSS_TIME_SIZE);

    if (ret < DSS_TIME_SIZE)

        return ret < 0 ? ret : AVERROR_EOF;



    sscanf(string, "%2d%2d%2d%2d%2d%2d", &y, &month, &d, &h, &minute, &sec);

    /* We deal with a two-digit year here, so set the default date to 2000

     * and hope it will never be used in the next century. */

    snprintf(datetime, sizeof(datetime), "%.4d-%.2d-%.2dT%.2d:%.2d:%.2d",

             y + 2000, month, d, h, minute, sec);

    return av_dict_set(&s->metadata, key, datetime, 0);

}
