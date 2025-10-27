static char *regname(uint32_t addr)

{

    static char buf[16];

    if (addr < PCI_IO_SIZE) {

        const char *r = reg[addr / 4];

        if (r != 0) {

            sprintf(buf, "%s+%u", r, addr % 4);

        } else {

            sprintf(buf, "0x%02x", addr);

        }

    } else {

        sprintf(buf, "??? 0x%08x", addr);

    }

    return buf;

}
