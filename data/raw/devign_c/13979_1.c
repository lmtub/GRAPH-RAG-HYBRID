static uint16_t eepro100_read2(EEPRO100State * s, uint32_t addr)

{

    uint16_t val;

    if (addr <= sizeof(s->mem) - sizeof(val)) {

        memcpy(&val, &s->mem[addr], sizeof(val));

    }



    switch (addr) {

    case SCBStatus:

    case SCBCmd:

        TRACE(OTHER, logout("addr=%s val=0x%04x\n", regname(addr), val));

        break;

    case SCBeeprom:

        val = eepro100_read_eeprom(s);

        TRACE(OTHER, logout("addr=%s val=0x%04x\n", regname(addr), val));

        break;

    default:

        logout("addr=%s val=0x%04x\n", regname(addr), val);

        missing("unknown word read");

    }

    return val;

}
