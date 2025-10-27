static void unassigned_mem_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

#ifdef DEBUG_UNASSIGNED

    printf("Unassigned mem write " TARGET_FMT_plx " = 0x%x\n", addr, val);

#endif

#if defined(TARGET_ALPHA) || defined(TARGET_SPARC) || defined(TARGET_MICROBLAZE)

    do_unassigned_access(addr, 1, 0, 0, 4);

#endif

}
