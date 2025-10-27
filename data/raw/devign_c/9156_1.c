static void restart_coroutine(void *opaque)

{

    Coroutine *co = opaque;



    DPRINTF("co=%p", co);



    qemu_coroutine_enter(co, NULL);

}
