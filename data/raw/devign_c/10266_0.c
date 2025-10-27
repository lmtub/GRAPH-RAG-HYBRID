void qemu_coroutine_delete(Coroutine *co_)

{

    CoroutineThreadState *s = coroutine_get_thread_state();

    CoroutineUContext *co = DO_UPCAST(CoroutineUContext, base, co_);



    if (s->pool_size < POOL_MAX_SIZE) {

        QLIST_INSERT_HEAD(&s->pool, &co->base, pool_next);

        co->base.caller = NULL;

        s->pool_size++;

        return;

    }



    g_free(co->stack);

    g_free(co);

}
