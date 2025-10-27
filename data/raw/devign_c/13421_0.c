static void spapr_io_write(void *opaque, hwaddr addr,

                           uint64_t data, unsigned size)

{

    switch (size) {

    case 1:

        cpu_outb(addr, data);

        return;

    case 2:

        cpu_outw(addr, data);

        return;

    case 4:

        cpu_outl(addr, data);

        return;

    }

    assert(0);

}
