void fork_start(void)

{


    mmap_fork_start();

    qemu_mutex_lock(&tb_ctx.tb_lock);

    cpu_list_lock();

}