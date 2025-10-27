static void unassigned_mem_write(void *opaque, target_phys_addr_t addr,

                                 uint64_t val, unsigned size)

{

#ifdef DEBUG_UNASSIGNED

    printf("Unassigned mem write " TARGET_FMT_plx " = 0x%"PRIx64"\n", addr, val);

#endif

#if defined(TARGET_ALPHA) || defined(TARGET_SPARC) || defined(TARGET_MICROBLAZE)

    cpu_unassigned_access(cpu_single_env, addr, 1, 0, 0, size);

#endif

}
