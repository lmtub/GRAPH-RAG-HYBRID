static int mxf_timestamp_to_str(uint64_t timestamp, char **str)
{
    struct tm time = { 0 };
    time.tm_year = (timestamp >> 48) - 1900;
    time.tm_mon  = (timestamp >> 40 & 0xFF) - 1;
    time.tm_mday = (timestamp >> 32 & 0xFF);
    time.tm_hour = (timestamp >> 24 & 0xFF);
    time.tm_min  = (timestamp >> 16 & 0xFF);
    time.tm_sec  = (timestamp >> 8  & 0xFF);
    *str = av_mallocz(32);
    if (!*str)
        return AVERROR(ENOMEM);
    strftime(*str, 32, "%Y-%m-%d %H:%M:%S", &time);
    return 0;
}