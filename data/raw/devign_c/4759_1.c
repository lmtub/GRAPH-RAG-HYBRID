static void test_parse_path_subprocess(void)

{

    /* All these should work without issue */

    qemu_set_log_filename("/tmp/qemu.log");

    qemu_set_log_filename("/tmp/qemu-%d.log");

    qemu_set_log_filename("/tmp/qemu.log.%d");

}
