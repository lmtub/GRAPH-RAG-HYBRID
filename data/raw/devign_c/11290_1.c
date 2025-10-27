static int yuv4_write_header(AVFormatContext *s)
{
    int* first_pkt = s->priv_data;
    if (s->nb_streams != 1)
        return AVERROR(EIO);
    if (s->streams[0]->codec->pix_fmt == PIX_FMT_YUV411P) {
        av_log(s, AV_LOG_ERROR, "Warning: generating rarely used 4:1:1 YUV stream, some mjpegtools might not work.\n");
    else if ((s->streams[0]->codec->pix_fmt != PIX_FMT_YUV420P) &&
             (s->streams[0]->codec->pix_fmt != PIX_FMT_YUV422P) &&
             (s->streams[0]->codec->pix_fmt != PIX_FMT_GRAY8) &&
             (s->streams[0]->codec->pix_fmt != PIX_FMT_YUV444P)) {
        av_log(s, AV_LOG_ERROR, "ERROR: yuv4mpeg only handles yuv444p, yuv422p, yuv420p, yuv411p and gray pixel formats. Use -pix_fmt to select one.\n");
        return AVERROR(EIO);
    *first_pkt = 1;
    return 0;