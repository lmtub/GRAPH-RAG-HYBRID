static void close_file(OutputStream *os)

{

    int64_t pos = avio_tell(os->out);

    avio_seek(os->out, 0, SEEK_SET);

    avio_wb32(os->out, pos);

    avio_flush(os->out);

    avio_close(os->out);

    os->out = NULL;

}
