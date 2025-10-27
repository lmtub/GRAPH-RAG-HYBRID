static void coroutine_delete(Coroutine *co)

{

    if (CONFIG_COROUTINE_POOL) {

        qemu_mutex_lock(&pool_lock);

        if (pool_size < pool_max_size) {

            QSLIST_INSERT_HEAD(&pool, co, pool_next);

            co->caller = NULL;

            pool_size++;

            qemu_mutex_unlock(&pool_lock);

            return;

        }

        qemu_mutex_unlock(&pool_lock);

    }



    qemu_coroutine_delete(co);

}
