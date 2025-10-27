static int ftp_read(URLContext *h, unsigned char *buf, int size)

{

    FTPContext *s = h->priv_data;

    int read, err, retry_done = 0;



    av_dlog(h, "ftp protocol read %d bytes\n", size);

  retry:

    if (s->state == DISCONNECTED) {

        /* optimization */

        if (s->position >= s->filesize)

            return 0;

        if ((err = ftp_connect_data_connection(h)) < 0)

            return err;

    }

    if (s->state == READY) {

        if (s->position >= s->filesize)

            return 0;

        if ((err = ftp_retrieve(s)) < 0)

            return err;

    }

    if (s->conn_data && s->state == DOWNLOADING) {

        read = ffurl_read(s->conn_data, buf, size);

        if (read >= 0) {

            s->position += read;

            if (s->position >= s->filesize) {

                /* server will terminate, but keep current position to avoid madness */

                /* save position to restart from it */

                int64_t pos = s->position;

                if (ftp_abort(h) < 0) {

                    s->position = pos;

                    return AVERROR(EIO);

                }

                s->position = pos;

            }

        }

        if (read <= 0 && s->position < s->filesize && !h->is_streamed) {

            /* Server closed connection. Probably due to inactivity */

            int64_t pos = s->position;

            av_log(h, AV_LOG_INFO, "Reconnect to FTP server.\n");

            if ((err = ftp_abort(h)) < 0)

                return err;

            if ((err = ftp_seek(h, pos, SEEK_SET)) < 0) {

                av_log(h, AV_LOG_ERROR, "Position cannot be restored.\n");

                return err;

            }

            if (!retry_done) {

                retry_done = 1;

                goto retry;

            }

        }

        return read;

    }



    av_log(h, AV_LOG_DEBUG, "FTP read failed\n");

    return AVERROR(EIO);

}
