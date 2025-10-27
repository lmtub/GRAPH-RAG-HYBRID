int av_open_input_file(AVFormatContext **ic_ptr, const char *filename, 

                       AVInputFormat *fmt,

                       int buf_size,

                       AVFormatParameters *ap)

{

    AVFormatContext *ic = NULL;

    int err;

    char buf[PROBE_BUF_SIZE];

    AVProbeData probe_data, *pd = &probe_data;



    ic = av_mallocz(sizeof(AVFormatContext));

    if (!ic) {

        err = AVERROR_NOMEM;

        goto fail;

    }

    pstrcpy(ic->filename, sizeof(ic->filename), filename);

    pd->filename = ic->filename;

    pd->buf = buf;

    pd->buf_size = 0;



    if (!fmt) {

        /* guess format if no file can be opened  */

        fmt = probe_input_format(pd, 0);

    }



    /* if no file needed do not try to open one */

    if (!fmt || !(fmt->flags & AVFMT_NOFILE)) {

        if (url_fopen(&ic->pb, filename, URL_RDONLY) < 0) {

            err = AVERROR_IO;

            goto fail;

        }

        if (buf_size > 0) {

            url_setbufsize(&ic->pb, buf_size);

        }

        /* read probe data */

        pd->buf_size = get_buffer(&ic->pb, buf, PROBE_BUF_SIZE);

        url_fseek(&ic->pb, 0, SEEK_SET);

    }

    

    /* guess file format */

    if (!fmt) {

        fmt = probe_input_format(pd, 1);

    }



    /* if still no format found, error */

    if (!fmt) {

        err = AVERROR_NOFMT;

        goto fail;

    }

        

    ic->iformat = fmt;



    /* allocate private data */

    ic->priv_data = av_mallocz(fmt->priv_data_size);

    if (!ic->priv_data) {

        err = AVERROR_NOMEM;

        goto fail;

    }



    /* check filename in case of an image number is expected */

    if (ic->iformat->flags & AVFMT_NEEDNUMBER) {

        if (filename_number_test(ic->filename) < 0) { 

            err = AVERROR_NUMEXPECTED;

            goto fail1;

        }

    }

    

    err = ic->iformat->read_header(ic, ap);

    if (err < 0)

        goto fail1;

    *ic_ptr = ic;

    return 0;

 fail1:

    if (!(fmt->flags & AVFMT_NOFILE)) {

        url_fclose(&ic->pb);

    }

 fail:

    if (ic) {

        av_free(ic->priv_data);

    }

    av_free(ic);

    *ic_ptr = NULL;

    return err;

}
