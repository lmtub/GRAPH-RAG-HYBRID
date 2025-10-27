void clear_blocks_dcbz32_ppc(DCTELEM *blocks)

{

POWERPC_TBL_DECLARE(powerpc_clear_blocks_dcbz32, 1);

    register int misal = ((unsigned long)blocks & 0x00000010);

    register int i = 0;

POWERPC_TBL_START_COUNT(powerpc_clear_blocks_dcbz32, 1);

#if 1

    if (misal) {

      ((unsigned long*)blocks)[0] = 0L;

      ((unsigned long*)blocks)[1] = 0L;

      ((unsigned long*)blocks)[2] = 0L;

      ((unsigned long*)blocks)[3] = 0L;

      i += 16;

    }

    for ( ; i < sizeof(DCTELEM)*6*64 ; i += 32) {

      asm volatile("dcbz %0,%1" : : "b" (blocks), "r" (i) : "memory");

    }

    if (misal) {

      ((unsigned long*)blocks)[188] = 0L;

      ((unsigned long*)blocks)[189] = 0L;

      ((unsigned long*)blocks)[190] = 0L;

      ((unsigned long*)blocks)[191] = 0L;

      i += 16;

    }

#else

    memset(blocks, 0, sizeof(DCTELEM)*6*64);

#endif

POWERPC_TBL_STOP_COUNT(powerpc_clear_blocks_dcbz32, 1);

}
