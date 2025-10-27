void qemu_savevm_send_postcopy_advise(QEMUFile *f)

{

    uint64_t tmp[2];

    tmp[0] = cpu_to_be64(getpagesize());

    tmp[1] = cpu_to_be64(1ul << qemu_target_page_bits());



    trace_qemu_savevm_send_postcopy_advise();

    qemu_savevm_command_send(f, MIG_CMD_POSTCOPY_ADVISE, 16, (uint8_t *)tmp);

}
