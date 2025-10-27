void blk_mig_init(void)

{

    QSIMPLEQ_INIT(&block_mig_state.bmds_list);

    QSIMPLEQ_INIT(&block_mig_state.blk_list);

    qemu_mutex_init(&block_mig_state.lock);



    register_savevm_live(NULL, "block", 0, 1, &savevm_block_handlers,

                         &block_mig_state);

}
