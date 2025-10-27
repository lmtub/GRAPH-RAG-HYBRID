static void bdrv_co_io_em_complete(void *opaque, int ret)

{

    CoroutineIOCompletion *co = opaque;



    co->ret = ret;

    qemu_coroutine_enter(co->coroutine, NULL);

}
