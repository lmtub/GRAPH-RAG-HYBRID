void cpu_physical_memory_rw(target_phys_addr_t addr, uint8_t *buf,

                            int len, int is_write)

{

    return address_space_rw(&address_space_memory, addr, buf, len, is_write);

}
