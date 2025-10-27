static bool get_phys_addr_pmsav7(CPUARMState *env, uint32_t address,

                                 MMUAccessType access_type, ARMMMUIdx mmu_idx,

                                 hwaddr *phys_ptr, int *prot,

                                 ARMMMUFaultInfo *fi)

{

    ARMCPU *cpu = arm_env_get_cpu(env);

    int n;

    bool is_user = regime_is_user(env, mmu_idx);



    *phys_ptr = address;

    *prot = 0;



    if (regime_translation_disabled(env, mmu_idx) ||

        m_is_ppb_region(env, address)) {

        /* MPU disabled or M profile PPB access: use default memory map.

         * The other case which uses the default memory map in the

         * v7M ARM ARM pseudocode is exception vector reads from the vector

         * table. In QEMU those accesses are done in arm_v7m_load_vector(),

         * which always does a direct read using address_space_ldl(), rather

         * than going via this function, so we don't need to check that here.

         */

        get_phys_addr_pmsav7_default(env, mmu_idx, address, prot);

    } else { /* MPU enabled */

        for (n = (int)cpu->pmsav7_dregion - 1; n >= 0; n--) {

            /* region search */

            uint32_t base = env->pmsav7.drbar[n];

            uint32_t rsize = extract32(env->pmsav7.drsr[n], 1, 5);

            uint32_t rmask;

            bool srdis = false;



            if (!(env->pmsav7.drsr[n] & 0x1)) {

                continue;




            if (!rsize) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "DRSR[%d]: Rsize field cannot be 0\n", n);

                continue;


            rsize++;

            rmask = (1ull << rsize) - 1;



            if (base & rmask) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "DRBAR[%d]: 0x%" PRIx32 " misaligned "

                              "to DRSR region size, mask = 0x%" PRIx32 "\n",

                              n, base, rmask);

                continue;




            if (address < base || address > base + rmask) {

                continue;




            /* Region matched */



            if (rsize >= 8) { /* no subregions for regions < 256 bytes */

                int i, snd;

                uint32_t srdis_mask;



                rsize -= 3; /* sub region size (power of 2) */

                snd = ((address - base) >> rsize) & 0x7;

                srdis = extract32(env->pmsav7.drsr[n], snd + 8, 1);



                srdis_mask = srdis ? 0x3 : 0x0;

                for (i = 2; i <= 8 && rsize < TARGET_PAGE_BITS; i *= 2) {

                    /* This will check in groups of 2, 4 and then 8, whether

                     * the subregion bits are consistent. rsize is incremented

                     * back up to give the region size, considering consistent

                     * adjacent subregions as one region. Stop testing if rsize

                     * is already big enough for an entire QEMU page.

                     */

                    int snd_rounded = snd & ~(i - 1);

                    uint32_t srdis_multi = extract32(env->pmsav7.drsr[n],

                                                     snd_rounded + 8, i);

                    if (srdis_mask ^ srdis_multi) {



                    srdis_mask = (srdis_mask << i) | srdis_mask;

                    rsize++;



            if (rsize < TARGET_PAGE_BITS) {

                qemu_log_mask(LOG_UNIMP,

                              "DRSR[%d]: No support for MPU (sub)region "

                              "alignment of %" PRIu32 " bits. Minimum is %d\n",

                              n, rsize, TARGET_PAGE_BITS);

                continue;


            if (srdis) {

                continue;






        if (n == -1) { /* no hits */

            if (!pmsav7_use_background_region(cpu, mmu_idx, is_user)) {

                /* background fault */

                fi->type = ARMFault_Background;

                return true;


            get_phys_addr_pmsav7_default(env, mmu_idx, address, prot);

        } else { /* a MPU hit! */

            uint32_t ap = extract32(env->pmsav7.dracr[n], 8, 3);

            uint32_t xn = extract32(env->pmsav7.dracr[n], 12, 1);



            if (m_is_system_region(env, address)) {

                /* System space is always execute never */

                xn = 1;




            if (is_user) { /* User mode AP bit decoding */

                switch (ap) {

                case 0:

                case 1:

                case 5:

                    break; /* no access */

                case 3:

                    *prot |= PAGE_WRITE;


                case 2:

                case 6:










                default:

                    qemu_log_mask(LOG_GUEST_ERROR,

                                  "DRACR[%d]: Bad value for AP bits: 0x%"

                                  PRIx32 "\n", n, ap);


            } else { /* Priv. mode AP bits decoding */

                switch (ap) {

                case 0:

                    break; /* no access */

                case 1:

                case 2:

                case 3:

                    *prot |= PAGE_WRITE;


                case 5:

                case 6:










                default:

                    qemu_log_mask(LOG_GUEST_ERROR,

                                  "DRACR[%d]: Bad value for AP bits: 0x%"

                                  PRIx32 "\n", n, ap);





            /* execute never */

            if (xn) {

                *prot &= ~PAGE_EXEC;






    fi->type = ARMFault_Permission;

    fi->level = 1;

    return !(*prot & (1 << access_type));
