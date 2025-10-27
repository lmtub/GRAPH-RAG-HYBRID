static uint64_t lan9118_16bit_mode_read(void *opaque, target_phys_addr_t offset,

                                        unsigned size)

{

    switch (size) {

    case 2:

        return lan9118_readw(opaque, offset);

    case 4:

        return lan9118_readl(opaque, offset, size);

    }



    hw_error("lan9118_read: Bad size 0x%x\n", size);

    return 0;

}
