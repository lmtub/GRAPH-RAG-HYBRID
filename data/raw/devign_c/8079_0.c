static void qvirtio_scsi_start(const char *extra_opts)

{

    char *cmdline;



    cmdline = g_strdup_printf(

                "-drive id=drv0,if=none,file=/dev/null,format=raw "

                "-device virtio-scsi-pci,id=vs0 "

                "-device scsi-hd,bus=vs0.0,drive=drv0 %s",

                extra_opts ? : "");

    qtest_start(cmdline);

    g_free(cmdline);

}
