static int get_physical_address(CPUState *env, target_phys_addr_t *physical,

                                int *prot, int *access_index,

                                target_ulong address, int rw, int mmu_idx)

{

    int is_user = mmu_idx == MMU_USER_IDX;



    if (rw == 2)

        return get_physical_address_code(env, physical, prot, address,

                                         is_user);

    else

        return get_physical_address_data(env, physical, prot, address, rw,

                                         is_user);

}
