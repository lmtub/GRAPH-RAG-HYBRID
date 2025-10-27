void qed_release(BDRVQEDState *s)

{

    aio_context_release(bdrv_get_aio_context(s->bs));

}
