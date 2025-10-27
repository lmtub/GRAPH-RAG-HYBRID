eeprom_t *eeprom93xx_new(DeviceState *dev, uint16_t nwords)

{

    /* Add a new EEPROM (with 16, 64 or 256 words). */

    eeprom_t *eeprom;

    uint8_t addrbits;



    switch (nwords) {

        case 16:

        case 64:

            addrbits = 6;

            break;

        case 128:

        case 256:

            addrbits = 8;

            break;

        default:

            assert(!"Unsupported EEPROM size, fallback to 64 words!");

            nwords = 64;

            addrbits = 6;

    }



    eeprom = (eeprom_t *)g_malloc0(sizeof(*eeprom) + nwords * 2);

    eeprom->size = nwords;

    eeprom->addrbits = addrbits;

    /* Output DO is tristate, read results in 1. */

    eeprom->eedo = 1;

    logout("eeprom = 0x%p, nwords = %u\n", eeprom, nwords);

    vmstate_register(dev, 0, &vmstate_eeprom, eeprom);

    return eeprom;

}
