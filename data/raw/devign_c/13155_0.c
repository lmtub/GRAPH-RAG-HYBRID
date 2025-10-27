static uint64_t eepro100_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    EEPRO100State *s = opaque;



    switch (size) {

    case 1: return eepro100_read1(s, addr);

    case 2: return eepro100_read2(s, addr);

    case 4: return eepro100_read4(s, addr);

    default: abort();

    }

}
