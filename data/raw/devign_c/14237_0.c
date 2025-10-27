static int64_t buffered_get_rate_limit(void *opaque)

{

    QEMUFileBuffered *s = opaque;

  

    return s->xfer_limit;

}
