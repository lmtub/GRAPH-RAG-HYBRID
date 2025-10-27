static void curl_readv_bh_cb(void *p)

{

    CURLState *state;

    int running;



    CURLAIOCB *acb = p;

    BDRVCURLState *s = acb->common.bs->opaque;



    qemu_bh_delete(acb->bh);

    acb->bh = NULL;



    size_t start = acb->sector_num * SECTOR_SIZE;

    size_t end;



    // In case we have the requested data already (e.g. read-ahead),

    // we can just call the callback and be done.

    switch (curl_find_buf(s, start, acb->nb_sectors * SECTOR_SIZE, acb)) {

        case FIND_RET_OK:

            qemu_aio_release(acb);

            // fall through

        case FIND_RET_WAIT:

            return;

        default:

            break;

    }



    // No cache found, so let's start a new request

    state = curl_init_state(s);

    if (!state) {

        acb->common.cb(acb->common.opaque, -EIO);

        qemu_aio_release(acb);

        return;

    }



    acb->start = 0;

    acb->end = (acb->nb_sectors * SECTOR_SIZE);



    state->buf_off = 0;

    g_free(state->orig_buf);

    state->buf_start = start;

    state->buf_len = acb->end + s->readahead_size;

    end = MIN(start + state->buf_len, s->len) - 1;

    state->orig_buf = g_malloc(state->buf_len);

    state->acb[0] = acb;



    snprintf(state->range, 127, "%zd-%zd", start, end);

    DPRINTF("CURL (AIO): Reading %d at %zd (%s)\n",

            (acb->nb_sectors * SECTOR_SIZE), start, state->range);

    curl_easy_setopt(state->curl, CURLOPT_RANGE, state->range);



    curl_multi_add_handle(s->multi, state->curl);



    /* Tell curl it needs to kick things off */

    curl_multi_socket_action(s->multi, CURL_SOCKET_TIMEOUT, 0, &running);

}
