static void bw_io_write(void *opaque, target_phys_addr_t addr,

                        uint64_t val, unsigned size)

{

    switch (size) {

    case 1:

        cpu_outb(addr, val);

        break;

    case 2:

        cpu_outw(addr, val);

        break;

    case 4:

        cpu_outl(addr, val);

        break;

    default:

        abort();

    }

}
