static uint32_t regtype_to_ss(uint8_t type)

{

    if (type & PCI_BASE_ADDRESS_MEM_TYPE_64) {

        return 3;

    }

    if (type == PCI_BASE_ADDRESS_SPACE_IO) {

        return 1;

    }

    return 2;

}
