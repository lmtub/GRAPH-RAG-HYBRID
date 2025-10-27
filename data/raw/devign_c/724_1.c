static void blk_mig_unlock(void)

{

    qemu_mutex_unlock(&block_mig_state.lock);

}
