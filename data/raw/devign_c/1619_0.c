int qemu_boot_set(const char *boot_order)

{

    if (!boot_set_handler) {

        return -EINVAL;

    }

    return boot_set_handler(boot_set_opaque, boot_order);

}
