static int ftp_open(URLContext *h, const char *url, int flags)

{

    char proto[10], path[MAX_URL_SIZE];

    int err;

    FTPContext *s = h->priv_data;



    av_dlog(h, "ftp protocol open\n");



    s->state = DISCONNECTED;

    s->filesize = -1;

    s->position = 0;



    av_url_split(proto, sizeof(proto),

                 s->credencials, sizeof(s->credencials),

                 s->hostname, sizeof(s->hostname),

                 &s->server_control_port,

                 path, sizeof(path),

                 url);



    if (s->server_control_port < 0 || s->server_control_port > 65535)

        s->server_control_port = 21;



    if ((err = ftp_connect_control_connection(h)) < 0)

        goto fail;



    if ((err = ftp_current_dir(s)) < 0)

        goto fail;

    av_strlcat(s->path, path, sizeof(s->path));



    if (ftp_restart(s, 0) < 0) {

        h->is_streamed = 1;

    } else {

        if (ftp_file_size(s) < 0 && flags & AVIO_FLAG_READ)

            h->is_streamed = 1;

        if (s->write_seekable != 1 && flags & AVIO_FLAG_WRITE)

            h->is_streamed = 1;

    }



    return 0;



  fail:

    av_log(h, AV_LOG_ERROR, "FTP open failed\n");

    ffurl_closep(&s->conn_control);

    ffurl_closep(&s->conn_data);

    return err;

}
