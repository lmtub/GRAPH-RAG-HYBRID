static int posix_aio_flush(void *opaque)

{

    PosixAioState *s = opaque;

    return !!s->first_aio;

}
