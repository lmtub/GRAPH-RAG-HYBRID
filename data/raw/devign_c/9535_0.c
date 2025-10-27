static void __attribute__((destructor)) coroutine_cleanup(void)

{

    Coroutine *co;

    Coroutine *tmp;



    QSLIST_FOREACH_SAFE(co, &pool, pool_next, tmp) {

        QSLIST_REMOVE_HEAD(&pool, pool_next);

        qemu_coroutine_delete(co);

    }

}
