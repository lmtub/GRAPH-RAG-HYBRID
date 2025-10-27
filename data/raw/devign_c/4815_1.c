static void colo_process_checkpoint(MigrationState *s)
{
    QIOChannelBuffer *bioc;
    QEMUFile *fb = NULL;
    int64_t current_time, checkpoint_time = qemu_clock_get_ms(QEMU_CLOCK_HOST);
    Error *local_err = NULL;
    int ret;
    failover_init_state();
    s->rp_state.from_dst_file = qemu_file_get_return_path(s->to_dst_file);
    if (!s->rp_state.from_dst_file) {
        error_report("Open QEMUFile from_dst_file failed");
    /*
     * Wait for Secondary finish loading VM states and enter COLO
     * restore.
     */
    colo_receive_check_message(s->rp_state.from_dst_file,
                       COLO_MESSAGE_CHECKPOINT_READY, &local_err);
    if (local_err) {
    bioc = qio_channel_buffer_new(COLO_BUFFER_BASE_SIZE);
    fb = qemu_fopen_channel_output(QIO_CHANNEL(bioc));
    object_unref(OBJECT(bioc));
    qemu_mutex_lock_iothread();
    vm_start();
    qemu_mutex_unlock_iothread();
    trace_colo_vm_state_change("stop", "run");
    while (s->state == MIGRATION_STATUS_COLO) {
        current_time = qemu_clock_get_ms(QEMU_CLOCK_HOST);
        if (current_time - checkpoint_time <
            s->parameters.x_checkpoint_delay) {
            int64_t delay_ms;
            delay_ms = s->parameters.x_checkpoint_delay -
                       (current_time - checkpoint_time);
            g_usleep(delay_ms * 1000);
        ret = colo_do_checkpoint_transaction(s, bioc, fb);
        if (ret < 0) {
        checkpoint_time = qemu_clock_get_ms(QEMU_CLOCK_HOST);
out:
    /* Throw the unreported error message after exited from loop */
    if (local_err) {
        error_report_err(local_err);
    if (fb) {
        qemu_fclose(fb);
    if (s->rp_state.from_dst_file) {
        qemu_fclose(s->rp_state.from_dst_file);