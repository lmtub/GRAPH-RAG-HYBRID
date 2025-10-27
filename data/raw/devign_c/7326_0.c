void fork_start(void)

{

    pthread_mutex_lock(&tcg_ctx.tb_ctx.tb_lock);

    pthread_mutex_lock(&exclusive_lock);

    mmap_fork_start();

}
