static void curl_setup_preadv(BlockDriverState *bs, CURLAIOCB *acb)

{

    CURLState *state;

    int running;



    BDRVCURLState *s = bs->opaque;



    uint64_t start = acb->offset;

    uint64_t end;



    qemu_mutex_lock(&s->mutex);



    // In case we have the requested data already (e.g. read-ahead),

    // we can just call the callback and be done.

    if (curl_find_buf(s, start, acb->bytes, acb)) {

        goto out;

    }



    // No cache found, so let's start a new request

    for (;;) {

        state = curl_find_state(s);

        if (state) {

            break;

        }

        qemu_mutex_unlock(&s->mutex);

        aio_poll(bdrv_get_aio_context(bs), true);

        qemu_mutex_lock(&s->mutex);

    }



    if (curl_init_state(s, state) < 0) {

        curl_clean_state(state);

        acb->ret = -EIO;

        goto out;

    }



    acb->start = 0;

    acb->end = MIN(acb->bytes, s->len - start);



    state->buf_off = 0;

    g_free(state->orig_buf);

    state->buf_start = start;

    state->buf_len = MIN(acb->end + s->readahead_size, s->len - start);

    end = start + state->buf_len - 1;

    state->orig_buf = g_try_malloc(state->buf_len);

    if (state->buf_len && state->orig_buf == NULL) {

        curl_clean_state(state);

        acb->ret = -ENOMEM;

        goto out;

    }

    state->acb[0] = acb;



    snprintf(state->range, 127, "%" PRIu64 "-%" PRIu64, start, end);

    DPRINTF("CURL (AIO): Reading %" PRIu64 " at %" PRIu64 " (%s)\n",

            acb->bytes, start, state->range);

    curl_easy_setopt(state->curl, CURLOPT_RANGE, state->range);



    curl_multi_add_handle(s->multi, state->curl);



    /* Tell curl it needs to kick things off */

    curl_multi_socket_action(s->multi, CURL_SOCKET_TIMEOUT, 0, &running);



out:

    qemu_mutex_unlock(&s->mutex);

}
