static void map_page(uint8_t **ptr, uint64_t addr, uint32_t wanted)

{

    target_phys_addr_t len = wanted;



    if (*ptr) {

        cpu_physical_memory_unmap(*ptr, len, 1, len);

    }



    *ptr = cpu_physical_memory_map(addr, &len, 1);

    if (len < wanted) {

        cpu_physical_memory_unmap(*ptr, len, 1, len);

        *ptr = NULL;

    }

}
