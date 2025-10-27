static inline int get_phys_addr(CPUState *env, uint32_t address,

                                int access_type, int is_user,

                                uint32_t *phys_ptr, int *prot)

{

    /* Fast Context Switch Extension.  */

    if (address < 0x02000000)

        address += env->cp15.c13_fcse;



    if ((env->cp15.c1_sys & 1) == 0) {

        /* MMU/MPU disabled.  */

        *phys_ptr = address;

        *prot = PAGE_READ | PAGE_WRITE;

        return 0;

    } else if (arm_feature(env, ARM_FEATURE_MPU)) {

	return get_phys_addr_mpu(env, address, access_type, is_user, phys_ptr,

				 prot);

    } else if (env->cp15.c1_sys & (1 << 23)) {

        return get_phys_addr_v6(env, address, access_type, is_user, phys_ptr,

                                prot);

    } else {

        return get_phys_addr_v5(env, address, access_type, is_user, phys_ptr,

                                prot);

    }

}
