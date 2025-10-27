static void bh_run_aio_completions(void *opaque)

{

    QEMUBH **bh = opaque;

    qemu_bh_delete(*bh);

    qemu_free(bh);

    qemu_aio_process_queue();

}
