int ffio_close_null_buf(AVIOContext *s)

{

    DynBuffer *d = s->opaque;

    int size;



    avio_flush(s);



    size = d->size;

    av_free(d);

    av_free(s);

    return size;

}
