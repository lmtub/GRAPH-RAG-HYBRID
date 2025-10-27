Coroutine *qemu_coroutine_create(CoroutineEntry *entry)

{

    Coroutine *co;



    co = QSLIST_FIRST(&pool);

    if (co) {

        QSLIST_REMOVE_HEAD(&pool, pool_next);

        pool_size--;

    } else {

        co = qemu_coroutine_new();

    }



    co->entry = entry;

    return co;

}
