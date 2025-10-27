int coroutine_fn bdrv_co_pdiscard(BlockDriverState *bs, int64_t offset,

                                  int bytes)

{

    BdrvTrackedRequest req;

    int max_pdiscard, ret;

    int head, tail, align;




        return -ENOMEDIUM;




    if (bdrv_has_readonly_bitmaps(bs)) {

        return -EPERM;




    ret = bdrv_check_byte_request(bs, offset, bytes);

    if (ret < 0) {

        return ret;

    } else if (bs->read_only) {

        return -EPERM;


    assert(!(bs->open_flags & BDRV_O_INACTIVE));



    /* Do nothing if disabled.  */

    if (!(bs->open_flags & BDRV_O_UNMAP)) {

        return 0;




    if (!bs->drv->bdrv_co_pdiscard && !bs->drv->bdrv_aio_pdiscard) {

        return 0;




    /* Discard is advisory, but some devices track and coalesce

     * unaligned requests, so we must pass everything down rather than

     * round here.  Still, most devices will just silently ignore

     * unaligned requests (by returning -ENOTSUP), so we must fragment

     * the request accordingly.  */

    align = MAX(bs->bl.pdiscard_alignment, bs->bl.request_alignment);

    assert(align % bs->bl.request_alignment == 0);

    head = offset % align;

    tail = (offset + bytes) % align;



    bdrv_inc_in_flight(bs);

    tracked_request_begin(&req, bs, offset, bytes, BDRV_TRACKED_DISCARD);



    ret = notifier_with_return_list_notify(&bs->before_write_notifiers, &req);

    if (ret < 0) {





    max_pdiscard = QEMU_ALIGN_DOWN(MIN_NON_ZERO(bs->bl.max_pdiscard, INT_MAX),

                                   align);

    assert(max_pdiscard >= bs->bl.request_alignment);



    while (bytes > 0) {

        int num = bytes;



        if (head) {

            /* Make small requests to get to alignment boundaries. */

            num = MIN(bytes, align - head);

            if (!QEMU_IS_ALIGNED(num, bs->bl.request_alignment)) {

                num %= bs->bl.request_alignment;


            head = (head + num) % align;

            assert(num < max_pdiscard);

        } else if (tail) {

            if (num > align) {

                /* Shorten the request to the last aligned cluster.  */

                num -= tail;

            } else if (!QEMU_IS_ALIGNED(tail, bs->bl.request_alignment) &&

                       tail > bs->bl.request_alignment) {

                tail %= bs->bl.request_alignment;

                num -= tail;



        /* limit request size */

        if (num > max_pdiscard) {

            num = max_pdiscard;








        if (bs->drv->bdrv_co_pdiscard) {

            ret = bs->drv->bdrv_co_pdiscard(bs, offset, num);

        } else {

            BlockAIOCB *acb;

            CoroutineIOCompletion co = {

                .coroutine = qemu_coroutine_self(),

            };



            acb = bs->drv->bdrv_aio_pdiscard(bs, offset, num,

                                             bdrv_co_io_em_complete, &co);

            if (acb == NULL) {

                ret = -EIO;


            } else {

                qemu_coroutine_yield();

                ret = co.ret;



        if (ret && ret != -ENOTSUP) {





        offset += num;

        bytes -= num;


    ret = 0;

out:

    atomic_inc(&bs->write_gen);

    bdrv_set_dirty(bs, req.offset, req.bytes);

    tracked_request_end(&req);

    bdrv_dec_in_flight(bs);

    return ret;
