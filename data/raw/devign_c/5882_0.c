static void mpc8544_guts_write(void *opaque, target_phys_addr_t addr,

                               uint64_t value, unsigned size)

{

    addr &= MPC8544_GUTS_MMIO_SIZE - 1;



    switch (addr) {

    case MPC8544_GUTS_ADDR_RSTCR:

        if (value & MPC8544_GUTS_RSTCR_RESET) {

            qemu_system_reset_request();

        }

        break;

    default:

        fprintf(stderr, "guts: Unknown register write: %x = %x\n",

                (int)addr, (unsigned)value);

        break;

    }

}
