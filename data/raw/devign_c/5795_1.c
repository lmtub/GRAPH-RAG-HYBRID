static void await_reference_mb_row(const H264Context *const h, H264Picture *ref,

                                   int mb_y)

{

    int ref_field         = ref->reference - 1;

    int ref_field_picture = ref->field_picture;

    int ref_height        = 16 * h->mb_height >> ref_field_picture;



    if (!HAVE_THREADS || !(h->avctx->active_thread_type & FF_THREAD_FRAME))

        return;



    /* FIXME: It can be safe to access mb stuff

     * even if pixels aren't deblocked yet. */



    ff_thread_await_progress(&ref->tf,

                             FFMIN(16 * mb_y >> ref_field_picture,

                                   ref_height - 1),

                             ref_field_picture && ref_field);

}
