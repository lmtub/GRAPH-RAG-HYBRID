static void curl_multi_read(void *arg)

{

    CURLState *s = (CURLState *)arg;



    aio_context_acquire(s->s->aio_context);

    curl_multi_do_locked(s);

    curl_multi_check_completion(s->s);

    aio_context_release(s->s->aio_context);

}
