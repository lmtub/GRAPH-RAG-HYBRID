void restore_boot_order(void *opaque)

{

    char *normal_boot_order = opaque;

    static int first = 1;



    /* Restore boot order and remove ourselves after the first boot */

    if (first) {

        first = 0;

        return;

    }



    qemu_boot_set(normal_boot_order, NULL);



    qemu_unregister_reset(restore_boot_order, normal_boot_order);

    g_free(normal_boot_order);

}
