static void realtime_init(void)

{

    if (enable_mlock) {

        if (os_mlock() < 0) {

            fprintf(stderr, "qemu: locking memory failed\n");

            exit(1);

        }

    }

}
