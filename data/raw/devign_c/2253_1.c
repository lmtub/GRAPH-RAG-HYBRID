static uint32_t eepro100_read4(EEPRO100State * s, uint32_t addr)

{

    uint32_t val;

    if (addr <= sizeof(s->mem) - sizeof(val)) {

        memcpy(&val, &s->mem[addr], sizeof(val));

    }



    switch (addr) {

    case SCBStatus:

        TRACE(OTHER, logout("addr=%s val=0x%08x\n", regname(addr), val));

        break;

    case SCBPointer:

#if 0

        val = eepro100_read_pointer(s);

#endif

        TRACE(OTHER, logout("addr=%s val=0x%08x\n", regname(addr), val));

        break;

    case SCBPort:

        val = eepro100_read_port(s);

        TRACE(OTHER, logout("addr=%s val=0x%08x\n", regname(addr), val));

        break;

    case SCBCtrlMDI:

        val = eepro100_read_mdi(s);

        break;

    default:

        logout("addr=%s val=0x%08x\n", regname(addr), val);

        missing("unknown longword read");

    }

    return val;

}
