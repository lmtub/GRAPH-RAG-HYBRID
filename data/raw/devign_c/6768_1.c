static BlockDriverAIOCB *curl_aio_readv(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    BDRVCURLState *s = bs->opaque;

    CURLAIOCB *acb;

    size_t start = sector_num * SECTOR_SIZE;

    size_t end;

    CURLState *state;



    acb = qemu_aio_get(&curl_aio_pool, bs, cb, opaque);

    if (!acb)

        return NULL;



    acb->qiov = qiov;



    // In case we have the requested data already (e.g. read-ahead),

    // we can just call the callback and be done.



    switch (curl_find_buf(s, start, nb_sectors * SECTOR_SIZE, acb)) {

        case FIND_RET_OK:

            qemu_aio_release(acb);

            // fall through

        case FIND_RET_WAIT:

            return &acb->common;

        default:

            break;

    }



    // No cache found, so let's start a new request



    state = curl_init_state(s);

    if (!state)

        return NULL;



    acb->start = 0;

    acb->end = (nb_sectors * SECTOR_SIZE);



    state->buf_off = 0;

    if (state->orig_buf)

        qemu_free(state->orig_buf);

    state->buf_start = start;

    state->buf_len = acb->end + READ_AHEAD_SIZE;

    end = MIN(start + state->buf_len, s->len) - 1;

    state->orig_buf = qemu_malloc(state->buf_len);

    state->acb[0] = acb;



    snprintf(state->range, 127, "%lld-%lld", (long long)start, (long long)end);

    dprintf("CURL (AIO): Reading %d at %lld (%s)\n", (nb_sectors * SECTOR_SIZE), start, state->range);

    curl_easy_setopt(state->curl, CURLOPT_RANGE, state->range);



    curl_multi_add_handle(s->multi, state->curl);

    curl_multi_do(s);



    return &acb->common;

}
