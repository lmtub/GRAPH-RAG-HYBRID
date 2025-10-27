static void mips_qemu_write (void *opaque, target_phys_addr_t addr,

                             uint64_t val, unsigned size)

{

    if ((addr & 0xffff) == 0 && val == 42)

        qemu_system_reset_request ();

    else if ((addr & 0xffff) == 4 && val == 42)

        qemu_system_shutdown_request ();

}
