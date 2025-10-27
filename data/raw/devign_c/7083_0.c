static void co_test_cb(void *opaque)

{

    WorkerTestData *data = opaque;



    active = 1;

    data->n = 0;

    data->ret = -EINPROGRESS;

    thread_pool_submit_co(worker_cb, data);



    /* The test continues in test_submit_co, after qemu_coroutine_enter... */



    g_assert_cmpint(data->n, ==, 1);

    data->ret = 0;

    active--;



    /* The test continues in test_submit_co, after qemu_aio_wait_all... */

}
