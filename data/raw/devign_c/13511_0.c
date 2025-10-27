void do_interrupt(int intno, int is_int, int error_code, 

                  unsigned int next_eip, int is_hw)

{

#ifdef DEBUG_PCALL

    if (loglevel) {

        static int count;

        fprintf(logfile, "%d: interrupt: vector=%02x error_code=%04x int=%d\n",

                count, intno, error_code, is_int);

        cpu_x86_dump_state(env, logfile, X86_DUMP_CCOP);

#if 0

        {

            int i;

            uint8_t *ptr;

            printf("       code=");

            ptr = env->segs[R_CS].base + env->eip;

            for(i = 0; i < 16; i++) {

                printf(" %02x", ldub(ptr + i));

            }

            printf("\n");

        }

#endif

        count++;

    }

#endif

    if (env->cr[0] & CR0_PE_MASK) {

        do_interrupt_protected(intno, is_int, error_code, next_eip, is_hw);

    } else {

        do_interrupt_real(intno, is_int, error_code, next_eip);

    }

}
