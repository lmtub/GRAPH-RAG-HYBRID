void address_space_read(AddressSpace *as, target_phys_addr_t addr, uint8_t *buf, int len)

{

    address_space_rw(as, addr, buf, len, false);

}
