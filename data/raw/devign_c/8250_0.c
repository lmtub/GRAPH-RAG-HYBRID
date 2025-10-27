static void ecc_mem_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    printf("ECC: Unsupported write 0x" TARGET_FMT_plx " %02x\n",

           addr, val & 0xff);

}
