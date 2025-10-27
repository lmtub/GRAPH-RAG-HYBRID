void helper_memalign(uint32_t addr, uint32_t dr, uint32_t wr, uint32_t size)

{

    uint32_t mask;



    switch (size) {

        case 4: mask = 3; break;

        case 2: mask = 1; break;

        default:

        case 1: mask = 0; break;

    }



    if (addr & mask) {

            qemu_log("unaligned access addr=%x size=%d, wr=%d\n",

                     addr, size, wr);

            if (!(env->sregs[SR_MSR] & MSR_EE)) {

                return;

            }



            env->sregs[SR_ESR] = ESR_EC_UNALIGNED_DATA | (wr << 10) \

                                 | (dr & 31) << 5;

            if (size == 4) {

                env->sregs[SR_ESR] |= 1 << 11;

            }

            helper_raise_exception(EXCP_HW_EXCP);

    }

}
