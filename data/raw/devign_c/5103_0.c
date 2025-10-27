static void coroutine_fn qed_need_check_timer_entry(void *opaque)

{

    BDRVQEDState *s = opaque;

    int ret;



    /* The timer should only fire when allocating writes have drained */

    assert(!s->allocating_acb);



    trace_qed_need_check_timer_cb(s);



    qed_acquire(s);

    qed_plug_allocating_write_reqs(s);



    /* Ensure writes are on disk before clearing flag */

    ret = bdrv_co_flush(s->bs->file->bs);

    qed_release(s);

    if (ret < 0) {

        qed_unplug_allocating_write_reqs(s);

        return;

    }



    s->header.features &= ~QED_F_NEED_CHECK;

    ret = qed_write_header(s);

    (void) ret;



    qed_unplug_allocating_write_reqs(s);



    ret = bdrv_co_flush(s->bs);

    (void) ret;

}
