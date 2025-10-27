static void curl_close(BlockDriverState *bs)

{

    BDRVCURLState *s = bs->opaque;



    DPRINTF("CURL: Close\n");

    curl_detach_aio_context(bs);

    qemu_mutex_destroy(&s->mutex);



    g_free(s->cookie);

    g_free(s->url);




}