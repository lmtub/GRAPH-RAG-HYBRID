static inline int put_dwords(uint32_t addr, uint32_t *buf, int num)

{

    int i;



    for(i = 0; i < num; i++, buf++, addr += sizeof(*buf)) {

        uint32_t tmp = cpu_to_le32(*buf);

        cpu_physical_memory_rw(addr,(uint8_t *)&tmp, sizeof(tmp), 1);

    }



    return 1;

}
