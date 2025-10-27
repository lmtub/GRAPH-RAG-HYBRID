static void coroutine_fn bdrv_create_co_entry(void *opaque)

{

    Error *local_err = NULL;

    int ret;



    CreateCo *cco = opaque;

    assert(cco->drv);



    ret = cco->drv->bdrv_create(cco->filename, cco->opts, &local_err);

    if (local_err) {

        error_propagate(&cco->err, local_err);

    }

    cco->ret = ret;

}
